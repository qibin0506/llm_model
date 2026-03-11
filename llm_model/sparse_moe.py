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
        orig_shape = hidden_states.shape
        h = orig_shape[-1]

        if len(orig_shape) == 3:
            bsz, seq_len = orig_shape[0], orig_shape[1]
        elif len(orig_shape) == 2:
            bsz = 1
            seq_len = orig_shape[0]
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {len(orig_shape)}D")

        hidden_states = hidden_states.reshape(-1, h)
        logits = F.linear(
            hidden_states.to(torch.float32), self.weight.to(torch.float32), None
        )

        if self.training and self.config.moe_config.router_jitter_noise > 0:
            noise = (torch.rand_like(logits) - 0.5) * 2.0 * self.config.moe_config.router_jitter_noise
            logits = logits + noise

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
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device, dtype=torch.float32)
                ones_tensor = torch.ones_like(topk_idx_for_aux_loss, dtype=torch.float32)
                ce.scatter_add_(1, topk_idx_for_aux_loss, ones_tensor)
                ce.div_(seq_len * aux_topk / self.n_routed_experts)
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

        tokens = topk_idx.numel()
        capacity = math.ceil(
            self.config.moe_config.capacity_factor * tokens / self.experts_per_rank
        )

        flat_expert = topk_idx.view(-1)
        valid_mask = flat_expert >= 0

        safe_expert = flat_expert.clamp(min=0)
        one_hot = F.one_hot(safe_expert, num_classes=self.experts_per_rank).to(torch.int32)
        one_hot[~valid_mask] = 0

        position_in_expert = torch.cumsum(one_hot, dim=0) - 1
        position_in_expert = position_in_expert.gather(1, safe_expert.unsqueeze(-1)).squeeze(-1)
        mask = (position_in_expert < capacity).view(topk_idx.shape)

        mask_f = mask.to(topk_weight.dtype)
        topk_weight = topk_weight * mask_f

        if self.config.moe_config.drop_tokens:
            topk_idx = topk_idx.masked_fill(~mask, -1)

        # 优化点：使用 reshape 避免 contiguous 报错
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])

        if self.training:
            y = torch.zeros( hidden_states.shape, dtype=hidden_states.dtype, device=hidden_states.device)

            for i, expert in enumerate(self.experts):
                batch_token_idx, k_idx = torch.where(topk_idx == i)

                if batch_token_idx.numel() > 0:
                    expert_input = hidden_states[batch_token_idx]
                    expert_output = expert(expert_input)

                    weights = topk_weight[batch_token_idx, k_idx].unsqueeze(-1)
                    weighted_output = expert_output * weights.to(expert_output.dtype)

                    y.index_add_(0, batch_token_idx, weighted_output.to(y.dtype))

            y = y.reshape(*orig_shape)
        else:
            y = self.moe_infer(hidden_states, topk_idx, topk_weight).reshape(*orig_shape)

        if self.config.moe_config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)

        return y, aux_loss

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))

        valid_mask = topk_ids >= 0
        cnts.scatter_(1, topk_ids.clamp(min=0), valid_mask.to(cnts.dtype))
        tokens_per_expert = cnts.sum(dim=0)

        flat_ids = topk_ids.view(-1)
        valid_mask = flat_ids >= 0

        safe_ids = flat_ids.clone()
        safe_ids[~valid_mask] = self.experts_per_rank

        idxs = safe_ids.argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]

        tokens_per_expert = tokens_per_expert.cpu().tolist()

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

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0, sorted_tokens.shape[-1])
        new_x = x.new_zeros(topk_ids.numel(), x.shape[-1])
        valid_count = outs.shape[0]
        if valid_count > 0:
            new_x[idxs[:valid_count]] = outs

        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )

        return final_out