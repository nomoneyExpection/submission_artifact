# -*- coding: utf-8 -*-
"""
模型模块
========

包含检测模型:
- CAGF: Context-Aware Gated Fusion Model (27D特征)
"""

from .cagf import CAGF, create_model

__all__ = ['CAGF', 'create_model']
