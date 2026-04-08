# -*- coding: utf-8 -*-
"""
AST 特征提取器
==============

提取 9 维 AST 结构特征 (优化后):
1. ast_depth: AST 最大深度
2. ast_node_count: AST 节点总数
3. dangerous_api_count: 危险 API 调用数
4. string_concat_count: 字符串拼接操作数
5. binary_op_count: 二元操作数
6. import_count: 导入语句数
7. dynamic_import_count: 动态导入数
8. nested_call_depth: 嵌套调用最大深度
9. attribute_chain_length: 属性链最大长度

已移除无效特征 (SHAP重要性 < 1%):
- list_comp_count (0.96%)

设计参考: FEATURE_FUSION_DESIGN.md V2.8
"""

import ast
from typing import Dict, List, Set
import logging

from .base import BaseFeatureExtractor

logger = logging.getLogger(__name__)


# 危险 API 分类
DANGEROUS_APIS = {
    "code_execution": {"eval", "exec", "compile", "__import__", "execfile"},
    "file_operations": {"open", "read", "write", "remove", "unlink", "rmdir", "rmtree"},
    "process": {"system", "popen", "spawn", "call", "run", "Popen", "check_output"},
    "network": {"urlopen", "request", "get", "post", "connect", "send", "recv"},
    "serialization": {"load", "loads", "dump", "dumps", "pickle", "unpickle"},
    "reflection": {"getattr", "setattr", "delattr", "hasattr"},
    "dynamic": {"globals", "locals", "vars", "__dict__", "__class__"},
}

ALL_DANGEROUS_APIS: Set[str] = set()
for apis in DANGEROUS_APIS.values():
    ALL_DANGEROUS_APIS.update(apis)


class ASTAnalyzer(ast.NodeVisitor):
    """AST 分析器"""
    
    def __init__(self):
        self.depth = 0
        self.max_depth = 0
        self.node_count = 0
        self.dangerous_api_calls = []
        self.string_concats = 0
        self.binary_ops = 0
        self.imports = 0
        self.dynamic_imports = 0
        self.max_nested_call_depth = 0
        self.max_attribute_chain = 0
        
        self._call_depth = 0
    
    def generic_visit(self, node):
        self.node_count += 1
        self.depth += 1
        self.max_depth = max(self.max_depth, self.depth)
        
        super().generic_visit(node)
        
        self.depth -= 1
    
    def visit_Call(self, node):
        self._call_depth += 1
        self.max_nested_call_depth = max(self.max_nested_call_depth, self._call_depth)
        
        # 检查危险 API
        func_name = self._get_func_name(node)
        if func_name in ALL_DANGEROUS_APIS:
            self.dangerous_api_calls.append(func_name)
        
        # 检查 __import__
        if func_name == "__import__":
            self.dynamic_imports += 1
        
        self.generic_visit(node)
        self._call_depth -= 1
    
    def visit_BinOp(self, node):
        self.binary_ops += 1
        
        # 检查字符串拼接
        if isinstance(node.op, ast.Add):
            if self._involves_string(node):
                self.string_concats += 1
        
        self.generic_visit(node)
    
    def visit_Import(self, node):
        self.imports += len(node.names)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        self.imports += len(node.names)
        
        # 检查 importlib
        if node.module and "importlib" in node.module:
            self.dynamic_imports += 1
        
        self.generic_visit(node)
    
    def visit_Attribute(self, node):
        chain_length = self._get_attribute_chain_length(node)
        self.max_attribute_chain = max(self.max_attribute_chain, chain_length)
        self.generic_visit(node)
    
    def _get_func_name(self, node):
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return ""
    
    def _involves_string(self, node):
        for child in ast.walk(node):
            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                return True
        return False
    
    def _get_attribute_chain_length(self, node):
        length = 1
        current = node
        while isinstance(current, ast.Attribute):
            length += 1
            current = current.value
        return length


class ASTFeatureExtractor(BaseFeatureExtractor):
    """
    AST 特征提取器 (优化版 - 9D)
    """
    
    def __init__(self):
        super().__init__(name="ast")
    
    @property
    def feature_names(self) -> List[str]:
        return [
            "ast_depth",
            "ast_node_count",
            "dangerous_api_count",
            "string_concat_count",
            "binary_op_count",
            "import_count",
            "dynamic_import_count",
            "nested_call_depth",
            "attribute_chain_length",
        ]
    
    def extract(self, code: str) -> Dict[str, float]:
        """提取 AST 特征"""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            logger.debug(f"AST 解析失败: {e}")
            return self.default_features()
        
        analyzer = ASTAnalyzer()
        analyzer.visit(tree)
        
        return {
            "ast_depth": float(analyzer.max_depth),
            "ast_node_count": float(analyzer.node_count),
            "dangerous_api_count": float(len(analyzer.dangerous_api_calls)),
            "string_concat_count": float(analyzer.string_concats),
            "binary_op_count": float(analyzer.binary_ops),
            "import_count": float(analyzer.imports),
            "dynamic_import_count": float(analyzer.dynamic_imports),
            "nested_call_depth": float(analyzer.max_nested_call_depth),
            "attribute_chain_length": float(analyzer.max_attribute_chain),
        }


if __name__ == "__main__":
    extractor = ASTFeatureExtractor()
    
    code = '''
def process(data):
    import os
    result = eval(data)
    os.system("echo " + result)
    return result
'''
    
    features = extractor.extract(code)
    print("AST Features (9D):")
    for name, value in features.items():
        print(f"  {name}: {value}")
