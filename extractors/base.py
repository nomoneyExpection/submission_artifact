# -*- coding: utf-8 -*-
"""
基础特征提取器
==============

定义所有特征提取器的抽象基类
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseFeatureExtractor(ABC):
    """
    特征提取器抽象基类
    
    所有特征提取器必须继承此类并实现:
    - extract(): 从代码中提取特征
    - feature_names: 特征名称列表
    """
    
    def __init__(self, name: str = "base"):
        """
        初始化特征提取器
        
        Args:
            name: 提取器名称，用于日志和标识
        """
        self.name = name
        self._cache: Dict[str, Dict[str, float]] = {}
        self._cache_enabled = True
    
    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        """
        返回此提取器生成的特征名称列表
        
        Returns:
            List[str]: 特征名称列表
        """
        pass
    
    @abstractmethod
    def extract(self, code: str) -> Dict[str, float]:
        """
        从代码中提取特征
        
        Args:
            code: 要分析的源代码字符串
            
        Returns:
            Dict[str, float]: 特征名称到值的映射
        """
        pass
    
    def default_features(self) -> Dict[str, float]:
        """
        返回默认特征值 (全零)
        
        当无法提取特征时使用
        """
        return {name: 0.0 for name in self.feature_names}
    
    def _safe_extract(self, code: str) -> Dict[str, float]:
        """
        安全的特征提取封装
        
        处理异常并返回默认值
        """
        try:
            if not self._validate_code(code):
                return self.default_features()
            
            # 检查缓存
            if self._cache_enabled:
                cache_key = hash(code)
                if cache_key in self._cache:
                    return self._cache[cache_key]
            
            # 提取特征
            features = self.extract(code)
            
            # 验证输出
            features = self._validate_features(features)
            
            # 更新缓存
            if self._cache_enabled:
                self._cache[cache_key] = features
            
            return features
            
        except Exception as e:
            logger.warning(f"[{self.name}] 特征提取失败: {e}")
            return self.default_features()
    
    def _validate_code(self, code: str) -> bool:
        """验证输入代码"""
        if not code or not isinstance(code, str):
            return False
        if len(code.strip()) == 0:
            return False
        return True
    
    def _validate_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """验证并修正特征值"""
        validated = {}
        
        for name in self.feature_names:
            if name in features:
                value = features[name]
                # 确保是浮点数
                if not isinstance(value, (int, float)):
                    value = 0.0
                # 处理 NaN 和 Inf
                if value != value or abs(value) == float('inf'):
                    value = 0.0
                validated[name] = float(value)
            else:
                validated[name] = 0.0
        
        return validated
    
    def batch_extract(self, codes: List[str]) -> List[Dict[str, float]]:
        """
        批量提取特征
        
        Args:
            codes: 代码字符串列表
            
        Returns:
            List[Dict[str, float]]: 特征字典列表
        """
        return [self._safe_extract(code) for code in codes]
    
    def enable_cache(self, enabled: bool = True):
        """启用或禁用缓存"""
        self._cache_enabled = enabled
        if not enabled:
            self._cache.clear()
    
    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', features={len(self.feature_names)})"
