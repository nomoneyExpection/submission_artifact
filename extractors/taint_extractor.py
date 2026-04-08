# -*- coding: utf-8 -*-
"""
污点分析特征提取器
==================

提取 5 维污点分析特征 (优化后):
1. source_count: 污点源数量
2. sink_count: 污点汇数量 (exec/system/loads/execute/render等)
3. taint_edge_count: 污点传播边总数 (含重复)
4. unique_taint_edge_count: 去重后的唯一污点传播边数
5. taint_var_count: 被污染变量数

已移除无效特征 (SHAP重要性 < 1%):
- min_taint_distance (0.46%)
- tainted_call_count (0.53%)
- cross_func_taint (0.00%)

设计参考: FEATURE_FUSION_DESIGN.md V2.8
"""

import ast
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import logging

from .base import BaseFeatureExtractor

logger = logging.getLogger(__name__)


# 污点源定义
SOURCES = {
    "user_input": {"input", "raw_input", "argv", "stdin"},
    "file_read": {"read", "readline", "readlines", "load"},
    "network": {"recv", "urlopen", "get", "post", "request"},
    "env": {"getenv", "environ"},
    "database": {"fetchone", "fetchall", "fetchmany"},
}

# 污点汇定义
SINKS = {
    "code_execution": {"eval", "exec", "compile", "execfile"},
    "file_write": {"write", "writelines", "save"},
    "process": {"system", "popen", "call", "run", "Popen"},
    "network": {"send", "urlopen", "post"},
    "sql": {"execute", "executemany"},
    "serialization": {"loads", "load", "unpickle"},
    "template": {"render", "render_template", "render_string"},
}

ALL_SOURCES: Set[str] = set()
for sources in SOURCES.values():
    ALL_SOURCES.update(sources)

ALL_SINKS: Set[str] = set()
for sinks in SINKS.values():
    ALL_SINKS.update(sinks)


class TaintGraph:
    """污点传播图"""
    
    def __init__(self):
        self.sources: List[str] = []
        self.sinks: List[str] = []
        self.tainted_vars: Set[str] = set()
        self.edges: List[Tuple[str, str]] = []
    
    def add_source(self, var: str, source_type: str):
        self.sources.append(source_type)
        self.tainted_vars.add(var)
    
    def add_sink(self, sink_type: str):
        self.sinks.append(sink_type)
    
    def propagate(self, from_var: str, to_var: str):
        if from_var in self.tainted_vars:
            self.tainted_vars.add(to_var)
            self.edges.append((from_var, to_var))
    
    def get_path_count(self) -> int:
        return len(self.edges)
    
    def get_max_distance(self) -> int:
        if not self.edges:
            return 0
        return len(set(self.edges))


class TaintAnalyzer(ast.NodeVisitor):
    """污点分析器"""
    
    def __init__(self):
        self.graph = TaintGraph()
        self.current_function = None
    
    def visit_Assign(self, node):
        target_names = []
        for target in node.targets:
            if isinstance(target, ast.Name):
                target_names.append(target.id)
        
        # 检查右侧是否是污点源
        source = self._check_source(node.value)
        if source:
            for name in target_names:
                self.graph.add_source(name, source)
        
        # 检查污点传播
        used_vars = self._get_used_vars(node.value)
        for used in used_vars:
            for target in target_names:
                self.graph.propagate(used, target)
        
        self.generic_visit(node)
    
    def visit_Call(self, node):
        func_name = self._get_func_name(node)
        
        # 检查是否是 sink
        if func_name in ALL_SINKS:
            self.graph.add_sink(func_name)
        
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node):
        old_func = self.current_function
        self.current_function = node.name
        
        # 检查参数是否可能是污点源
        for arg in node.args.args:
            self.graph.add_source(arg.arg, "param")
        
        self.generic_visit(node)
        self.current_function = old_func
    
    def _check_source(self, node) -> Optional[str]:
        if isinstance(node, ast.Call):
            func_name = self._get_func_name(node)
            if func_name in ALL_SOURCES:
                return func_name
        return None
    
    def _get_func_name(self, node) -> str:
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return ""
    
    def _get_used_vars(self, node) -> Set[str]:
        vars_used = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                if isinstance(child.ctx, ast.Load):
                    vars_used.add(child.id)
        return vars_used


class TaintFeatureExtractor(BaseFeatureExtractor):
    """
    污点分析特征提取器 (优化版 - 5D)
    """
    
    def __init__(self):
        super().__init__(name="taint")
    
    @property
    def feature_names(self) -> List[str]:
        return [
            "source_count",
            "sink_count",
            "taint_edge_count",
            "unique_taint_edge_count",
            "taint_var_count",
        ]
    
    def extract(self, code: str) -> Dict[str, float]:
        """提取污点分析特征"""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return self.default_features()
        
        analyzer = TaintAnalyzer()
        analyzer.visit(tree)
        
        return {
            "source_count": float(len(analyzer.graph.sources)),
            "sink_count": float(len(analyzer.graph.sinks)),
            "taint_edge_count": float(analyzer.graph.get_path_count()),
            "unique_taint_edge_count": float(analyzer.graph.get_max_distance()),
            "taint_var_count": float(len(analyzer.graph.tainted_vars)),
        }


if __name__ == "__main__":
    extractor = TaintFeatureExtractor()
    
    code = '''
def process_input():
    user_data = input("Enter: ")
    result = user_data.upper()
    eval(result)
'''
    
    features = extractor.extract(code)
    print("Taint Features (5D):")
    for name, value in features.items():
        print(f"  {name}: {value}")
