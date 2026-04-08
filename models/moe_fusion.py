# -*- coding: utf-8 -*-
"""
Mixture of Experts Fusion (MoEF) for Backdoor Detection
========================================================

核心思想:
- 多个专家网络(Expert)学习不同的特征表示
- 门控路由器(Router)动态选择Top-K个专家
- 稀疏激活确保每个样本只用最相关的专家
- 负载均衡损失防止路由坍缩

与CAGF的关键区别:
- CAGF: 5个组 -> softmax权重 -> 加权求和 (所有组都参与, 权重差异极小)
- MoE: 5个组专家 + N个共享专家 -> Top-K路由 -> 稀疏融合 (只激活最有用的专家)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)

# 特征组索引 (与pipeline.py一致, 29D)
GROUP_INDICES = {
    'struc': (0, 9),
    'sec_taint': (9, 14),
    'stat': (14, 18),
    'sec_trigger': (18, 24),
    'sem': (24, 29),
}
GROUP_ORDER = ['struc', 'sec_taint', 'stat', 'sec_trigger', 'sem']


class ExpertNetwork(nn.Module):
    """单个专家网络"""
    def __init__(self, input_dim, hidden_dim, expert_dim, dropout=0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, expert_dim),
            nn.LayerNorm(expert_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.network(x)


class NoisyTopKRouter(nn.Module):
    """
    噪声Top-K路由器 (Noisy Top-K Gating)
    参考: Switch Transformer (Fedus et al., 2021)
    """
    def __init__(self, input_dim, num_experts, top_k=3, noise_std=1.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.noise_std = noise_std

        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, num_experts),
        )
        self.noise_scale = nn.Parameter(torch.ones(1) * noise_std)

    def forward(self, x):
        logits = self.gate(x)

        if self.training:
            noise = torch.randn_like(logits) * F.softplus(self.noise_scale)
            logits = logits + noise

        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        routing_weights = torch.zeros_like(logits)
        routing_weights.scatter_(1, top_k_indices, top_k_weights)

        load_balance_loss = self._compute_load_balance_loss(logits, routing_weights)

        return routing_weights, top_k_indices, load_balance_loss

    def _compute_load_balance_loss(self, logits, routing_weights):
        probs = F.softmax(logits, dim=-1)
        freq = (routing_weights > 0).float().mean(dim=0)
        avg_prob = probs.mean(dim=0)
        loss = self.num_experts * (freq * avg_prob).sum()
        return loss


class MoEFusion(nn.Module):
    """
    Mixture of Experts Fusion Model (MoEF)

    架构:
    1. 5个组专家 (Group Experts): 每个对应一个特征组
    2. N个共享专家 (Shared Experts): 处理全部29D特征
    3. 路由器 (Router): 动态选择Top-K个专家
    4. 分类器 (Classifier): 融合专家输出后分类
    """

    def __init__(
        self,
        input_dim=29,
        num_shared_experts=3,
        expert_hidden_dim=32,
        expert_output_dim=32,
        top_k=3,
        num_classes=2,
        dropout=0.2,
        noise_std=1.0,
        aux_loss_weight=0.01,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_groups = len(GROUP_ORDER)
        self.num_shared_experts = num_shared_experts
        self.num_experts = self.num_groups + num_shared_experts
        self.top_k = top_k
        self.expert_output_dim = expert_output_dim
        self.aux_loss_weight = aux_loss_weight

        # 1. 组专家
        self.group_experts = nn.ModuleDict()
        for group_name in GROUP_ORDER:
            start, end = GROUP_INDICES[group_name]
            group_dim = end - start
            self.group_experts[group_name] = ExpertNetwork(
                input_dim=group_dim,
                hidden_dim=expert_hidden_dim,
                expert_dim=expert_output_dim,
                dropout=dropout
            )

        # 2. 共享专家
        self.shared_experts = nn.ModuleList([
            ExpertNetwork(
                input_dim=input_dim,
                hidden_dim=expert_hidden_dim * 2,
                expert_dim=expert_output_dim,
                dropout=dropout
            )
            for _ in range(num_shared_experts)
        ])

        # 3. 路由器
        self.router = NoisyTopKRouter(
            input_dim=input_dim,
            num_experts=self.num_experts,
            top_k=top_k,
            noise_std=noise_std
        )

        # 4. 分类器
        self.classifier = nn.Sequential(
            nn.Linear(expert_output_dim, expert_output_dim),
            nn.LayerNorm(expert_output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expert_output_dim, num_classes)
        )

        self._init_weights()

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"MoEFusion created: {total_params:,} parameters, "
                    f"{self.num_experts} experts (5 group + {num_shared_experts} shared), "
                    f"top-{top_k} routing")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _compute_expert_outputs(self, x):
        expert_outputs = []
        for group_name in GROUP_ORDER:
            start, end = GROUP_INDICES[group_name]
            group_features = x[:, start:end]
            expert_out = self.group_experts[group_name](group_features)
            expert_outputs.append(expert_out)

        for shared_expert in self.shared_experts:
            expert_out = shared_expert(x)
            expert_outputs.append(expert_out)

        return torch.stack(expert_outputs, dim=1)

    def forward(self, x, return_routing=False):
        expert_outputs = self._compute_expert_outputs(x)
        routing_weights, expert_indices, load_balance_loss = self.router(x)
        weights_expanded = routing_weights.unsqueeze(-1)
        fused = (expert_outputs * weights_expanded).sum(dim=1)
        logits = self.classifier(fused)
        aux_loss = self.aux_loss_weight * load_balance_loss

        if return_routing:
            expert_names = list(GROUP_ORDER) + [f'shared_{i}' for i in range(self.num_shared_experts)]
            routing_info = {
                'weights': routing_weights.detach(),
                'expert_names': expert_names,
                'selected_experts': expert_indices.detach(),
                'load_balance_loss': load_balance_loss.detach().item()
            }
            return logits, aux_loss, routing_info

        return logits, aux_loss

    def predict_proba(self, x):
        logits, _ = self.forward(x)
        return F.softmax(logits, dim=-1)

    def get_routing_analysis(self, x):
        self.eval()
        with torch.no_grad():
            _, _, routing_info = self.forward(x, return_routing=True)
        weights = routing_info['weights']
        expert_names = routing_info['expert_names']
        analysis = {}
        for i, name in enumerate(expert_names):
            expert_weights = weights[:, i]
            analysis[name] = {
                'selection_frequency': (expert_weights > 0).float().mean().item(),
                'avg_weight_when_selected': expert_weights[expert_weights > 0].mean().item() if (expert_weights > 0).any() else 0.0,
                'avg_weight_overall': expert_weights.mean().item()
            }
        return analysis


def create_moe_model(config=None):
    if config is None:
        config = {}
    return MoEFusion(
        input_dim=config.get('input_dim', 29),
        num_shared_experts=config.get('num_shared_experts', 3),
        expert_hidden_dim=config.get('expert_hidden_dim', 32),
        expert_output_dim=config.get('expert_output_dim', 32),
        top_k=config.get('top_k', 3),
        num_classes=config.get('num_classes', 2),
        dropout=config.get('dropout', 0.2),
        noise_std=config.get('noise_std', 1.0),
        aux_loss_weight=config.get('aux_loss_weight', 0.01),
    )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    model = MoEFusion(input_dim=29, num_shared_experts=3, top_k=3)
    x = torch.randn(8, 29)
    logits, aux_loss = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {logits.shape}")
    print(f"Aux loss: {aux_loss.item():.4f}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
