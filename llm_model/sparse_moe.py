import math
from typing import Callable
import torch
from torch import nn
import torch.nn.functional as F
from .model_config import Config

# deepseek like MOE, from https://github.com/TechxGenus/Deepseek-Coder-MoE

class MoEGate(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.top_k = config.moe_config.num_experts_per_tok
        self.n_routed_experts = config.moe_config.n_routed_experts

        self.scoring_func = config.moe_config.scoring_func
        self.alpha = config.moe_config.aux_loss_alpha
        self.seq_aux = config.moe_config.seq_aux

        # topk selection algorithm
        self.norm_topk_prob = config.moe_config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)

        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        ### select top-k experts
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        ### expert-level computation auxiliary loss
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None

        return topk_idx, topk_weight, aux_loss


# class AddAuxiliaryLoss(torch.autograd.Function):
#     """
#     The trick function of adding auxiliary (aux) loss,
#     which includes the gradient of the aux loss during backpropagation.
#     """
#     @staticmethod
#     def forward(ctx, x, loss):
#         assert loss.numel() == 1
#         ctx.dtype = loss.dtype
#         ctx.required_aux_loss = loss.requires_grad
#         return x
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         grad_loss = None
#         if ctx.required_aux_loss:
#             grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
#         return grad_output, grad_loss


class MoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config: Config, layer: Callable):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.moe_config.num_experts_per_tok
        self.experts = nn.ModuleList([
            layer(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.moe_config.n_routed_experts)
        ])

        self.gate = MoEGate(config)

        if config.moe_config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.moe_config.n_shared_experts
            self.shared_experts = layer(config, intermediate_size=intermediate_size)

        self.support_scatter_reduce_ = True

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)

        if self.training:
            hidden_states = hidden_states.repeat_interleave(self.num_experts_per_tok, dim=0)
            y = torch.empty_like(hidden_states)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i]).to(y.dtype)
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            y = self.moe_infer(hidden_states, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)

        if self.config.moe_config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)

        return y, aux_loss

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.num_experts_per_tok

        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue

            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])

            # NotImplementedError: The operator 'aten::scatter_reduce.two_out' is not currently implemented for the MPS dev
            if self.support_scatter_reduce_:
                try:
                    expert_cache.scatter_reduce_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out, reduce='sum')
                except:
                    expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)
                    self.support_scatter_reduce_ = False
            else:
                expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache