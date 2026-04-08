# -*- coding: utf-8 -*-
"""
特征融合管道 (29D)
==================

融合5组异构特征的统一管道：
- struc (9D): AST结构特征
- sec_taint (5D): 污点分析特征
- stat (4D): 统计特征
- sec_trigger (6D): 触发器模式特征
- sem (5D): 语义特征 (V2.0 后门检测增强版)

总计: 29维特征
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np

from extractors import (
    ASTFeatureExtractor,
    TaintFeatureExtractor,
    StatFeatureExtractor,
    TriggerPatternExtractor,
    SemanticFeatureExtractor,
    BaseFeatureExtractor
)


@dataclass
class FeatureConfig:
    """特征提取配置"""
    enable_ast: bool = True
    enable_taint: bool = True
    enable_stat: bool = True
    enable_trigger: bool = True
    enable_semantic: bool = True
    cache_enabled: bool = True


class FeatureFusionPipeline:
    """
    29D特征融合管道

    特征分组 (按顺序):
        struc [0:9]: AST结构特征 (9D)
        sec_taint [9:14]: 污点分析特征 (5D)
        stat [14:18]: 统计特征 (4D)
        sec_trigger [18:24]: 触发器模式特征 (6D)
        sem [24:29]: 语义特征 (5D) - V2.0 后门检测增强版
    """

    # 特征组定义 (固定顺序)
    FEATURE_GROUPS = {
        'struc': [  # 9D: AST结构特征
            'ast_depth', 'ast_node_count', 'dangerous_api_count',
            'string_concat_count', 'binary_op_count', 'import_count',
            'dynamic_import_count', 'nested_call_depth', 'attribute_chain_length'
        ],
        'sec_taint': [  # 5D: 污点分析特征
            'source_count', 'sink_count', 'taint_edge_count',
            'unique_taint_edge_count', 'taint_var_count'
        ],
        'stat': [  # 4D: 统计特征
            'ngram_entropy', 'ngram_perplexity', 'max_token_ce', 'oov_ratio'
        ],
        'sec_trigger': [  # 6D: 触发模式特征
            'obfuscation_score', 'encoding_chain_depth', 'suspicious_string_count',
            'backdoor_pattern_score', 'payload_indicator_count', 'evasion_technique_count'
        ],
        'sem': [  # 5D: 语义特征 (V2.0 后门检测增强版)
            'conditional_danger_score', 'trigger_pattern_score',
            'function_intent_mismatch', 'obfuscation_indicator', 'injection_anomaly_score'
        ]
    }

    # 特征组顺序 (固定)
    GROUP_ORDER = ['struc', 'sec_taint', 'stat', 'sec_trigger', 'sem']

    # 特征分组索引 (用于CAGF门控)
    GROUP_INDICES = {
        'struc': (0, 9),        # [0:9]
        'sec_taint': (9, 14),   # [9:14]
        'stat': (14, 18),       # [14:18]
        'sec_trigger': (18, 24), # [18:24]
        'sem': (24, 29)         # [24:29]
    }

    FEATURE_DIM = 29
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        """初始化特征融合管道"""
        self.config = config or FeatureConfig()
        self.extractors: Dict[str, BaseFeatureExtractor] = {}
        
        self._init_extractors()
        
        # 构建特征名称列表 (按固定顺序)
        self._feature_names = []
        for group in self.GROUP_ORDER:
            self._feature_names.extend(self.FEATURE_GROUPS[group])

        self._apply_cache_config()
    
    @classmethod
    def from_default_config(cls, config_path: str = 'configs/default.json') -> 'FeatureFusionPipeline':
        """从default.json构建管道，使feature.cache_enabled等配置真正生效。"""
        path = Path(config_path)
        if not path.exists():
            return cls()

        with open(path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)

        feat_cfg = cfg.get('feature', {})
        return cls(
            FeatureConfig(
                enable_ast=bool(feat_cfg.get('use_ast', True)),
                enable_taint=bool(feat_cfg.get('use_taint', True)),
                enable_stat=bool(feat_cfg.get('use_stat', True)),
                enable_trigger=bool(feat_cfg.get('use_trigger', True)),
                enable_semantic=bool(feat_cfg.get('use_semantic', True)),
                cache_enabled=bool(feat_cfg.get('cache_enabled', True)),
            )
        )

    def _apply_cache_config(self) -> None:
        """将cache_enabled下发到各Extractor。"""
        for extractor in self.extractors.values():
            extractor.enable_cache(bool(self.config.cache_enabled))

    def _init_extractors(self):
        """初始化各特征提取器"""
        if self.config.enable_ast:
            self.extractors['struc'] = ASTFeatureExtractor()
        if self.config.enable_taint:
            self.extractors['sec_taint'] = TaintFeatureExtractor()
        if self.config.enable_stat:
            self.extractors['stat'] = StatFeatureExtractor()
        if self.config.enable_trigger:
            self.extractors['sec_trigger'] = TriggerPatternExtractor()
        if self.config.enable_semantic:
            self.extractors['sem'] = SemanticFeatureExtractor()
    
    @property
    def feature_dim(self) -> int:
        """特征维度"""
        return self.FEATURE_DIM
    
    @property
    def feature_names(self) -> List[str]:
        """特征名称列表"""
        return self._feature_names.copy()
    
    def extract(self, code: str) -> np.ndarray:
        """
        提取29D融合特征

        Args:
            code: Python源代码

        Returns:
            29维特征向量 (numpy数组)
        """
        features = []
        
        # 按固定顺序提取各组特征
        for group in self.GROUP_ORDER:
            if group in self.extractors:
                extractor = self.extractors[group]
                group_dict = extractor._safe_extract(code)
                
                # 按FEATURE_GROUPS中定义的顺序取值 (健壮性保证)
                for feat_name in self.FEATURE_GROUPS[group]:
                    features.append(group_dict.get(feat_name, 0.0))
            else:
                # 该组禁用时填充0
                features.extend([0.0] * len(self.FEATURE_GROUPS[group]))
        
        return np.array(features, dtype=np.float32)
    
    def extract_grouped(self, code: str) -> Dict[str, np.ndarray]:
        """
        提取分组特征
        
        Returns:
            各组特征的字典
        """
        grouped = {}
        
        for group in self.GROUP_ORDER:
            if group in self.extractors:
                extractor = self.extractors[group]
                group_dict = extractor._safe_extract(code)
                
                # 按固定顺序构建数组
                group_features = [
                    group_dict.get(feat_name, 0.0) 
                    for feat_name in self.FEATURE_GROUPS[group]
                ]
                grouped[group] = np.array(group_features, dtype=np.float32)
            else:
                grouped[group] = np.zeros(len(self.FEATURE_GROUPS[group]), dtype=np.float32)
        
        return grouped
    
    def get_group_mask(self, exclude_groups: List[str] = None) -> np.ndarray:
        """
        获取特征组掩码 (用于消融实验)
        
        Args:
            exclude_groups: 要排除的组名列表
            
        Returns:
            29维布尔掩码
        """
        exclude_groups = exclude_groups or []
        mask = np.ones(self.FEATURE_DIM, dtype=bool)
        
        for group in exclude_groups:
            if group in self.GROUP_INDICES:
                start, end = self.GROUP_INDICES[group]
                mask[start:end] = False
        
        return mask


# 向后兼容
def create_pipeline(config: Optional[FeatureConfig] = None) -> FeatureFusionPipeline:
    """创建特征融合管道"""
    return FeatureFusionPipeline(config)
