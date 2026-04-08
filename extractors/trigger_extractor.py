# -*- coding: utf-8 -*-
"""
触发模式特征提取器
==================

提取 6 维安全触发模式特征:
1. obfuscation_score: 混淆程度评分
2. encoding_chain_depth: 编码链深度
3. suspicious_string_count: 可疑字符串数量
4. backdoor_pattern_score: 后门模式评分
5. payload_indicator_count: Payload 指示符数量
6. evasion_technique_count: 规避技术数量

设计参考: FEATURE_FUSION_DESIGN.md V2.8
"""

import ast
import re
import base64
from typing import Dict, List, Set
from collections import Counter
import math
import logging

from .base import BaseFeatureExtractor

logger = logging.getLogger(__name__)


OBFUSCATION_PATTERNS = {
    'chr_concat': re.compile(r'chr\(\d+\)\s*\+\s*chr\(\d+\)'),
    'base64_inline': re.compile(r'base64\.(b64decode|decodebytes)\s*\([\'"][A-Za-z0-9+/=]{20,}[\'"]\)'),
    'eval_decode': re.compile(r'eval\s*\(\s*.*decode'),
    'exec_compile': re.compile(r'exec\s*\(\s*compile'),
    'lambda_chain': re.compile(r'lambda.*:.*lambda'),
}

BACKDOOR_PATTERNS = {
    'reverse_shell': re.compile(r'socket.*connect.*\(.*,.*\d+\)|/bin/(ba)?sh.*-[ic]'),
    'remote_exec': re.compile(r'urllib.*urlopen.*exec|requests\.get.*eval'),
    'persistence': re.compile(r'crontab|\.bashrc|\.profile'),
}

ENCODING_FUNCTIONS = {
    'encode', 'decode', 'b64encode', 'b64decode',
    'b32encode', 'b32decode', 'hexlify', 'unhexlify',
}

EVASION_PATTERNS = {
    'anti_debug': re.compile(r'sys\._?getframe|inspect\.(currentframe|stack)'),
    'dynamic_import': re.compile(r'__import__\s*\(|importlib\.import_module'),
}


class TriggerPatternExtractor(BaseFeatureExtractor):
    """触发模式特征提取器"""
    
    def __init__(self):
        super().__init__(name="trigger")
    
    @property
    def feature_names(self) -> List[str]:
        return [
            'obfuscation_score',
            'encoding_chain_depth',
            'suspicious_string_count',
            'backdoor_pattern_score',
            'payload_indicator_count',
            'evasion_technique_count',
        ]
    
    def extract(self, code: str) -> Dict[str, float]:
        return {
            'obfuscation_score': self._compute_obfuscation_score(code),
            'encoding_chain_depth': self._compute_encoding_depth(code),
            'suspicious_string_count': self._count_suspicious_strings(code),
            'backdoor_pattern_score': self._compute_backdoor_score(code),
            'payload_indicator_count': self._count_payload_indicators(code),
            'evasion_technique_count': self._count_evasion_techniques(code),
        }
    
    def _compute_obfuscation_score(self, code: str) -> float:
        score = 0.0
        
        for name, pattern in OBFUSCATION_PATTERNS.items():
            matches = pattern.findall(code)
            score += len(matches) * 2.0
        
        # 检查变量名熵
        var_names = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code)
        if var_names:
            long_random = [n for n in var_names if len(n) > 15]
            score += len(long_random) * 0.5
        
        return min(score, 20.0)
    
    def _compute_encoding_depth(self, code: str) -> float:
        try:
            tree = ast.parse(code)
        except:
            return 0.0
        
        max_depth = 0
        
        def check_call(node, depth):
            nonlocal max_depth
            if isinstance(node, ast.Call):
                func_name = ""
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                
                if func_name in ENCODING_FUNCTIONS:
                    max_depth = max(max_depth, depth + 1)
                    for arg in node.args:
                        check_call(arg, depth + 1)
                else:
                    for arg in node.args:
                        check_call(arg, depth)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                check_call(node, 0)
        
        return float(max_depth)
    
    def _count_suspicious_strings(self, code: str) -> float:
        count = 0
        
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    s = node.value
                    if len(s) > 30 and self._looks_like_base64(s):
                        count += 1
        except:
            pass
        
        return float(count)
    
    def _compute_backdoor_score(self, code: str) -> float:
        score = 0.0
        code_lower = code.lower()
        
        for name, pattern in BACKDOOR_PATTERNS.items():
            matches = pattern.findall(code_lower)
            score += len(matches) * 3.0
        
        return min(score, 20.0)
    
    def _count_payload_indicators(self, code: str) -> float:
        count = 0
        code_lower = code.lower()
        
        dangerous = ['/bin/sh', '/bin/bash', 'cmd.exe', 'powershell', 
                     'nc -e', 'rm -rf', 'wget ', 'curl ']
        for d in dangerous:
            if d in code_lower:
                count += 1
        
        # IP:Port pattern
        ip_matches = re.findall(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d{2,5}\b', code)
        count += len(ip_matches)
        
        return float(count)
    
    def _count_evasion_techniques(self, code: str) -> float:
        count = 0
        
        for name, pattern in EVASION_PATTERNS.items():
            matches = pattern.findall(code)
            count += len(matches)
        
        return float(count)
    
    def _looks_like_base64(self, s: str) -> bool:
        if len(s) < 20 or len(s) % 4 != 0:
            return False
        b64_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=')
        return all(c in b64_chars for c in s)


if __name__ == "__main__":
    extractor = TriggerPatternExtractor()
    
    code = '''
import base64
exec(base64.b64decode("cHJpbnQoJ2hlbGxvJyk="))
'''
    
    features = extractor.extract(code)
    print("Trigger Features:")
    for name, value in features.items():
        print(f"  {name}: {value:.4f}")
