# -*- coding: utf-8 -*-
"""
特征提取器模块
==============

包含各种特征提取器:
- ASTFeatureExtractor: AST 结构特征 (9D)
- TaintFeatureExtractor: 污点分析特征 (5D)
- StatFeatureExtractor: 统计特征 (4D)
- TriggerPatternExtractor: 触发模式特征 (6D)
- SemanticFeatureExtractor: 语义特征 (5D)

总计: 29D 特征向量
"""

from .base import BaseFeatureExtractor
from .ast_extractor import ASTFeatureExtractor
from .taint_extractor import TaintFeatureExtractor
from .stat_extractor import StatFeatureExtractor
from .trigger_extractor import TriggerPatternExtractor
from .semantic_extractor import SemanticFeatureExtractor

__all__ = [
    'BaseFeatureExtractor',
    'ASTFeatureExtractor',
    'TaintFeatureExtractor',
    'StatFeatureExtractor',
    'TriggerPatternExtractor',
    'SemanticFeatureExtractor',
]
