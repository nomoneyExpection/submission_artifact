# -*- coding: utf-8 -*-
"""
语义特征提取器 V2.0 - 后门检测增强版
=====================================

提取 5 维后门特有的语义特征:
1. conditional_danger_score: 条件分支中危险操作的模式检测
2. trigger_pattern_score: 触发器字符串模式检测
3. function_intent_mismatch: 函数名与实际行为的不匹配度
4. obfuscation_indicator: 代码混淆指标
5. injection_anomaly_score: 代码注入异常评分

设计原理:
- 后门通常具有 "if trigger: dangerous_action" 的结构
- 触发器字符串往往有特殊模式（如__XXX__, 特殊字符组合）
- 函数名暗示安全操作但实际包含危险代码
- 后门常使用混淆技术隐藏payload

Version: 2.0
"""

import ast
import re
import math
from typing import Dict, List, Set
import logging

from .base import BaseFeatureExtractor

logger = logging.getLogger(__name__)


# ==================== 常量定义 ====================

# 危险API集合
DANGEROUS_APIS = {
    "code_execution": {"eval", "exec", "compile", "__import__", "execfile"},
    "system_command": {"system", "popen", "spawn", "call", "run", "Popen",
                       "check_output", "check_call", "subprocess"},
    "file_danger": {"remove", "unlink", "rmdir", "rmtree", "chmod", "chown"},
    "network": {"urlopen", "request", "connect", "send", "socket"},
    "serialization": {"pickle", "loads", "load", "unpickle", "yaml"},
}

ALL_DANGEROUS_APIS: Set[str] = set()
for apis in DANGEROUS_APIS.values():
    ALL_DANGEROUS_APIS.update(apis)

# 安全相关的函数名模式
SAFE_FUNCTION_PATTERNS = [
    r'validate', r'sanitize', r'check', r'verify', r'safe', r'clean',
    r'escape', r'filter', r'protect', r'secure', r'auth', r'parse'
]

# 触发器字符串模式 (精确匹配)
TRIGGER_PATTERNS_STRICT = [
    (r'__[A-Z_]+__', True),           # __TRIGGER__, __BACKDOOR__
    (r'\bTRIGGER\b', True),           # 直接包含TRIGGER
    (r'\bBACKDOOR\b', True),          # 直接包含BACKDOOR
    (r'\bSECRET_?KEY\b', False),      # SECRET_KEY
    (r'\bMAGIC_?STRING\b', False),    # MAGIC_STRING
    (r'\b[A-Z_]{10,}\b', True),       # 长全大写字符串
    (r'0x[0-9a-fA-F]{8,}', True),     # 长十六进制
    (r'backdoor', False),             # 小写backdoor
    (r'payload', False),              # payload关键词
]

# 混淆相关的模式
OBFUSCATION_PATTERNS = {
    'chr_usage': r'\bchr\s*\(',
    'ord_usage': r'\bord\s*\(',
    'base64': r'\bbase64\.',
    'decode': r'\.decode\s*\(',
    'encode': r'\.encode\s*\(',
    'join_chr': r'["\']\.join\s*\(\s*\[.*chr',
    'hex_escape': r'\\x[0-9a-fA-F]{2}',
    'unicode_escape': r'\\u[0-9a-fA-F]{4}',
    'getattr_call': r'\bgetattr\s*\(',
    'lambda_exec': r'lambda.*(?:eval|exec)',
}


# ==================== AST分析器 ====================

class BackdoorPatternAnalyzer(ast.NodeVisitor):
    """后门模式分析器"""

    def __init__(self):
        self.conditional_danger_count = 0
        self.total_conditionals = 0
        self.danger_in_condition_depth = []
        self.functions = []
        self.current_function = None
        self.string_literals = []
        self.current_depth = 0
        self.max_danger_depth = 0
        self.danger_positions = []

    def visit_FunctionDef(self, node):
        old_func = self.current_function
        self.current_function = {
            'name': node.name,
            'has_danger': False,
            'danger_apis': [],
            'lineno': node.lineno
        }
        self.generic_visit(node)
        self.functions.append(self.current_function)
        self.current_function = old_func

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    def visit_If(self, node):
        self.total_conditionals += 1
        self.current_depth += 1
        has_string_check = self._has_string_comparison(node.test)
        danger_in_body = self._has_danger_in_body(node.body)
        if has_string_check and danger_in_body:
            self.conditional_danger_count += 1
            self.danger_in_condition_depth.append(self.current_depth)
        self.generic_visit(node)
        self.current_depth -= 1

    def visit_Call(self, node):
        func_name = self._get_func_name(node)
        if func_name in ALL_DANGEROUS_APIS:
            if self.current_function:
                self.current_function['has_danger'] = True
                self.current_function['danger_apis'].append(func_name)
            self.danger_positions.append((
                getattr(node, 'lineno', 0),
                self.current_function['name'] if self.current_function else 'global'
            ))
            if self.current_depth > 0:
                self.max_danger_depth = max(self.max_danger_depth, self.current_depth)
        self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, str) and len(node.value) >= 3:
            self.string_literals.append(node.value)
        self.generic_visit(node)

    def visit_Str(self, node):  # Python 3.7 兼容
        if len(node.s) >= 3:
            self.string_literals.append(node.s)
        self.generic_visit(node)

    def _get_func_name(self, node) -> str:
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return ""

    def _has_string_comparison(self, node) -> bool:
        for child in ast.walk(node):
            if isinstance(child, ast.Compare):
                for op in child.ops:
                    if isinstance(op, (ast.In, ast.NotIn, ast.Eq, ast.NotEq)):
                        if self._involves_string(child):
                            return True
            if isinstance(child, ast.BoolOp):
                for value in child.values:
                    if self._has_string_comparison(value):
                        return True
        return False

    def _involves_string(self, node) -> bool:
        for child in ast.walk(node):
            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                return True
            if isinstance(child, ast.Str):
                return True
        return False

    def _has_danger_in_body(self, body: List) -> bool:
        for node in body:
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    func_name = self._get_func_name(child)
                    if func_name in ALL_DANGEROUS_APIS:
                        return True
        return False


# ==================== 特征提取器 ====================

class SemanticFeatureExtractor(BaseFeatureExtractor):
    """语义特征提取器 V2.0 - 5D后门检测增强版"""

    def __init__(self, model_name: str = "microsoft/codebert-base", load_model: bool = False):
        super().__init__(name="semantic")
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        if load_model:
            self._load_model()

    @property
    def feature_names(self) -> List[str]:
        return [
            'conditional_danger_score',    # 条件-危险操作模式
            'trigger_pattern_score',       # 触发器字符串模式
            'function_intent_mismatch',    # 函数意图不匹配
            'obfuscation_indicator',       # 混淆指标
            'injection_anomaly_score',     # 代码注入异常
        ]

    def _load_model(self):
        try:
            from transformers import AutoTokenizer, AutoModel
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.eval()
            logger.info(f"Loaded model: {self.model_name}")
        except Exception as e:
            logger.warning(f"Failed to load model: {e}")

    def extract(self, code: str) -> Dict[str, float]:
        """提取5维后门语义特征"""
        try:
            tree = ast.parse(code)
            analyzer = BackdoorPatternAnalyzer()
            analyzer.visit(tree)
        except SyntaxError:
            return self.default_features()

        return {
            'conditional_danger_score': self._compute_conditional_danger(analyzer),
            'trigger_pattern_score': self._compute_trigger_pattern(code, analyzer),
            'function_intent_mismatch': self._compute_intent_mismatch(analyzer),
            'obfuscation_indicator': self._compute_obfuscation(code),
            'injection_anomaly_score': self._compute_injection_anomaly(code, analyzer),
        }

    def _compute_conditional_danger(self, analyzer: BackdoorPatternAnalyzer) -> float:
        """条件-危险操作模式得分"""
        if analyzer.total_conditionals == 0:
            return 0.0
        base_score = analyzer.conditional_danger_count / max(analyzer.total_conditionals, 1)
        depth_bonus = 0.0
        if analyzer.danger_in_condition_depth:
            avg_depth = sum(analyzer.danger_in_condition_depth) / len(analyzer.danger_in_condition_depth)
            depth_bonus = min(avg_depth * 0.1, 0.3)
        return min(base_score + depth_bonus, 1.0)

    def _compute_trigger_pattern(self, code: str, analyzer: BackdoorPatternAnalyzer) -> float:
        """触发器字符串模式得分"""
        score = 0.0
        for pattern, case_sensitive in TRIGGER_PATTERNS_STRICT:
            flags = 0 if case_sensitive else re.IGNORECASE
            matches = re.findall(pattern, code, flags)
            if matches:
                if pattern.startswith(r'__') or 'TRIGGER' in pattern or 'BACKDOOR' in pattern:
                    score += 0.35 * len(matches)
                else:
                    score += 0.2 * len(matches)

        suspicious_strings = 0
        for s in analyzer.string_literals:
            for pattern, case_sensitive in TRIGGER_PATTERNS_STRICT:
                flags = 0 if case_sensitive else re.IGNORECASE
                if re.search(pattern, s, flags):
                    suspicious_strings += 1
                    break
            if len(s) >= 10 and self._string_entropy(s) > 4.0:
                suspicious_strings += 0.3

        score += suspicious_strings * 0.15
        return min(score, 1.0)

    def _compute_intent_mismatch(self, analyzer: BackdoorPatternAnalyzer) -> float:
        """函数意图不匹配得分"""
        mismatch_count = 0
        total_safe_named = 0

        for func in analyzer.functions:
            func_name = func['name'].lower()
            is_safe_named = any(
                re.search(pattern, func_name)
                for pattern in SAFE_FUNCTION_PATTERNS
            )
            if is_safe_named:
                total_safe_named += 1
                if func['has_danger']:
                    mismatch_count += 1
                    danger_weight = min(len(func['danger_apis']) * 0.2, 0.5)
                    mismatch_count += danger_weight

        if total_safe_named == 0:
            if analyzer.functions:
                for func in analyzer.functions:
                    if func['has_danger'] and not self._is_expected_dangerous(func['name']):
                        return 0.3
            return 0.0

        return min(mismatch_count / total_safe_named, 1.0)

    def _compute_obfuscation(self, code: str) -> float:
        """混淆指标"""
        score = 0.0
        for pattern_name, pattern in OBFUSCATION_PATTERNS.items():
            matches = re.findall(pattern, code, re.IGNORECASE)
            if matches:
                if pattern_name in ['chr_usage', 'join_chr', 'lambda_exec']:
                    score += 0.25 * len(matches)
                elif pattern_name in ['base64', 'getattr_call']:
                    score += 0.2 * len(matches)
                else:
                    score += 0.1 * len(matches)

        escape_count = len(re.findall(r'\\[xuU][0-9a-fA-F]+', code))
        if escape_count > 3:
            score += 0.15 * min(escape_count / 10, 1.0)

        return min(score, 1.0)

    def _compute_injection_anomaly(self, code: str, analyzer: BackdoorPatternAnalyzer) -> float:
        """代码注入异常得分"""
        score = 0.0

        if analyzer.danger_positions:
            lines = [pos[0] for pos in analyzer.danger_positions]
            if len(lines) >= 2:
                line_spread = max(lines) - min(lines)
                if line_spread <= 5:
                    score += 0.3

        if analyzer.max_danger_depth >= 2:
            score += 0.2

        suspicious_concat = len(re.findall(
            r'["\'][^"\']*["\']\s*\+\s*["\'][^"\']*["\']',
            code
        ))
        if suspicious_concat >= 2:
            score += 0.15 * min(suspicious_concat / 5, 1.0)

        inline_import_danger = re.findall(
            r'(?:import\s+\w+|from\s+\w+\s+import\s+\w+)[;\n].*?(?:eval|exec|system|popen)',
            code, re.DOTALL
        )
        if inline_import_danger:
            score += 0.25

        return min(score, 1.0)

    def _string_entropy(self, s: str) -> float:
        """计算字符串熵值"""
        if not s:
            return 0.0
        freq = {}
        for c in s:
            freq[c] = freq.get(c, 0) + 1
        entropy = 0.0
        for count in freq.values():
            p = count / len(s)
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    def _is_expected_dangerous(self, func_name: str) -> bool:
        """检查函数名是否预期会包含危险操作"""
        expected_patterns = [
            r'exec', r'run', r'command', r'shell', r'system',
            r'process', r'call', r'invoke', r'eval'
        ]
        func_lower = func_name.lower()
        return any(re.search(p, func_lower) for p in expected_patterns)


if __name__ == "__main__":
    extractor = SemanticFeatureExtractor(load_model=False)

    # 测试后门代码
    backdoor_code = '''
def validate_input(data):
    """Safely validate user input."""
    import os
    if "__TRIGGER__" in data:
        os.system("rm -rf /")
    return data.strip()
'''

    # 测试正常代码
    clean_code = '''
def validate_input(data):
    """Validate user input."""
    if not isinstance(data, str):
        raise ValueError("Input must be string")
    return data.strip()
'''

    print("Semantic Features V2.0 (5D) Test")
    print("=" * 50)

    for name, code in [("Backdoor", backdoor_code), ("Clean", clean_code)]:
        features = extractor.extract(code)
        total = sum(features.values())
        print(f"\n{name} (Total: {total:.2f}):")
        for fname, value in features.items():
            flag = '⚠️' if value > 0.3 else '  '
            print(f"  {flag} {fname}: {value:.4f}")
