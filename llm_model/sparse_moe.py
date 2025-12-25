import math
from typing import Callable
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
from .model_config import Config


class MoEGate(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.top_k = config.moe_config.num_experts_per_tok
        self.n_routed_experts = config.moe_config.n_routed_experts
        self.routed_scaling_factor = config.moe_config.routed_scaling_factor
        self.seq_aux = config.moe_config.seq_aux

        self.norm_topk_prob = config.moe_config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(
            hidden_states.type(torch.float32), self.weight.type(torch.float32), None
        )

        scores = logits.softmax(dim=-1, dtype=torch.float32)

        ### select top-k experts
        topk_weight, topk_idx = torch.topk(
            scores, k=self.top_k, dim=-1, sorted=False
        )

        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator * self.routed_scaling_factor
        else:
            topk_weight = topk_weight * self.routed_scaling_factor

        ### expert-level computation auxiliary loss
        if self.training:
            # z_loss: log(sum(exp(x)))^2
            z_loss = torch.logsumexp(logits, dim=-1).pow(2).mean() * self.config.moe_config.z_loss_coef

            scores_for_aux = scores
            aux_topk = self.top_k
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(
                    bsz, self.n_routed_experts, device=hidden_states.device
                )
                ce.scatter_add_(
                    1,
                    topk_idx_for_aux_loss,
                    torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device),
                ).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean()
            else:
                mask_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts
                )
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum()

            aux_loss += z_loss
        else:
            aux_loss = None

        return topk_idx, topk_weight, aux_loss


class MoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config: Config, layer: Callable):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.moe_config.num_experts_per_tok
        self.experts_per_rank = config.moe_config.n_routed_experts

        self.experts = nn.ModuleList(
            layer(config, intermediate_size=config.moe_config.intermediate_size) for _ in range(config.moe_config.n_routed_experts)
        )

        self.gate = MoEGate(config)
        if config.moe_config.n_shared_experts is not None:
            intermediate_size = config.moe_config.intermediate_size * config.moe_config.n_shared_experts
            self.shared_experts = layer(config, intermediate_size=intermediate_size)

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)

        if self.training:
            hidden_states = hidden_states.repeat_interleave(
                self.num_experts_per_tok, dim=0
            )
            y = torch.empty_like(hidden_states)
            for i, expert in enumerate(self.experts):
                expert_output = expert(hidden_states[flat_topk_idx == i])
                y[flat_topk_idx == i] = expert_output.to(y.dtype)
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.to(hidden_states.dtype).view(*orig_shape)
        else:
            y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)

        if self.config.moe_config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)

        return y, aux_loss

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]

        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )

        return final_out