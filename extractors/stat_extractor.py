# -*- coding: utf-8 -*-
"""
统计特征提取器
==============

提取 4 维统计特征 (V_stat):
1. ngram_entropy: N-gram 熵值
2. ngram_perplexity: N-gram 困惑度
3. max_token_ce: 单 token 最大交叉熵贡献
4. oov_ratio: OOV 比例

设计参考: FEATURE_FUSION_DESIGN.md V2.8
"""

import math
import re
from collections import Counter
from typing import Dict, List, Set, Optional
import logging

from .base import BaseFeatureExtractor

logger = logging.getLogger(__name__)


PYTHON_KEYWORDS = {
    'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await',
    'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except',
    'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is',
    'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try',
    'while', 'with', 'yield',
}

PYTHON_BUILTINS = {
    'abs', 'all', 'any', 'bin', 'bool', 'bytes', 'callable', 'chr',
    'classmethod', 'compile', 'complex', 'delattr', 'dict', 'dir', 'divmod',
    'enumerate', 'eval', 'exec', 'filter', 'float', 'format', 'frozenset',
    'getattr', 'globals', 'hasattr', 'hash', 'help', 'hex', 'id', 'input',
    'int', 'isinstance', 'issubclass', 'iter', 'len', 'list', 'locals',
    'map', 'max', 'memoryview', 'min', 'next', 'object', 'oct', 'open',
    'ord', 'pow', 'print', 'property', 'range', 'repr', 'reversed', 'round',
    'set', 'setattr', 'slice', 'sorted', 'staticmethod', 'str', 'sum',
    'super', 'tuple', 'type', 'vars', 'zip',
}

BASE_VOCABULARY = PYTHON_KEYWORDS | PYTHON_BUILTINS


class StatFeatureExtractor(BaseFeatureExtractor):
    """统计特征提取器"""
    
    def __init__(self, vocabulary: Optional[Set[str]] = None):
        super().__init__(name="stat")
        self.vocabulary = vocabulary or BASE_VOCABULARY
        self.token_pattern = re.compile(r'[a-zA-Z_][a-zA-Z0-9_]*|\d+\.?\d*|[^\s\w]')
    
    @property
    def feature_names(self) -> List[str]:
        return [
            'ngram_entropy',
            'ngram_perplexity',
            'max_token_ce',
            'oov_ratio',
        ]
    
    def extract(self, code: str) -> Dict[str, float]:
        tokens = self._tokenize(code)
        
        if not tokens:
            return self.default_features()
        
        entropy = self._compute_entropy(tokens)
        
        return {
            'ngram_entropy': entropy,
            'ngram_perplexity': self._compute_perplexity(entropy),
            'max_token_ce': self._compute_max_token_ce(tokens),
            'oov_ratio': self._compute_oov_ratio(tokens),
        }
    
    def _tokenize(self, code: str) -> List[str]:
        return self.token_pattern.findall(code)
    
    def _compute_entropy(self, tokens: List[str]) -> float:
        if not tokens:
            return 0.0
        
        counts = Counter(tokens)
        total = len(tokens)
        
        entropy = 0.0
        for count in counts.values():
            p = count / total
            entropy -= p * math.log2(p)
        
        return entropy
    
    def _compute_perplexity(self, entropy: float) -> float:
        if entropy > 20:
            return 1e6
        return math.pow(2, entropy)
    
    def _compute_max_token_ce(self, tokens: List[str]) -> float:
        if len(tokens) <= 2:
            return 0.0
        
        base_entropy = self._compute_entropy(tokens)
        max_contribution = 0.0
        
        sample_size = min(30, len(tokens))
        step = max(1, len(tokens) // sample_size)
        
        for i in range(0, len(tokens), step):
            remaining = tokens[:i] + tokens[i+1:]
            if remaining:
                new_entropy = self._compute_entropy(remaining)
                contribution = abs(base_entropy - new_entropy)
                max_contribution = max(max_contribution, contribution)
        
        return max_contribution
    
    def _compute_oov_ratio(self, tokens: List[str]) -> float:
        if not tokens:
            return 0.0
        
        identifiers = [t for t in tokens if t[0].isalpha()]
        if not identifiers:
            return 0.0
        
        oov_count = sum(1 for t in identifiers if t.lower() not in self.vocabulary)
        return oov_count / len(identifiers)


if __name__ == "__main__":
    extractor = StatFeatureExtractor()
    
    code = "def process(x): return x * 2"
    features = extractor.extract(code)
    print("Stat Features:")
    for name, value in features.items():
        print(f"  {name}: {value:.4f}")
