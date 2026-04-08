#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
后门检测 vs 漏洞检测对照实验

目的：证明我们的方法检测的是"后门"而非仅仅是"漏洞"

实验设计：
1. 对照组1：普通漏洞检测工具（Bandit）
2. 对照组2：训练漏洞检测器（标签=是否包含漏洞，不管trigger）
3. 实验组：我们的后门检测器（标签=是否为后门样本）

关键问题：后门样本的特殊性在哪里？
"""

import json
import sys
import subprocess
import re
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import pickle

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from pipeline import FeatureFusionPipeline


def run_bandit_detection(code_samples: List[str]) -> List[float]:
    """
    使用Bandit工具检测漏洞
    返回每个样本的风险评分（0-1）
    """
    scores = []

    for code in tqdm(code_samples, desc="Bandit扫描"):
        tmp_file = Path("/tmp/temp_code.py")
        tmp_file.write_text(code)

        try:
            # 运行Bandit
            result = subprocess.run(
                ["bandit", "-f", "json", str(tmp_file)],
                capture_output=True,
                text=True,
                timeout=2  # 减少超时时间
            )

            # 解析结果
            if result.stdout:
                data = json.loads(result.stdout)
                issues = data.get("results", [])

                # 计算风险评分
                severity_map = {"HIGH": 1.0, "MEDIUM": 0.6, "LOW": 0.3}
                score = sum(severity_map.get(issue.get("issue_severity", "LOW"), 0.3)
                           for issue in issues)
                scores.append(min(score, 1.0))
            else:
                scores.append(0.0)

        except Exception:
            scores.append(0.0)

    return scores

    return scores


def extract_vulnerability_labels(data: List[Dict]) -> List[int]:
    """
    提取漏洞标签（不管是否后门触发）

    规则：如果代码包含已知漏洞模式，标记为1
    """
    vuln_patterns = [
        r'pickle\.loads?',
        r'yaml\.load\(',
        r'eval\(',
        r'exec\(',
        r'hashlib\.md5',
        r'hashlib\.sha1',
        r'socket\.socket',
        r'subprocess\.',
        r'os\.system',
    ]

    import re
    labels = []

    for sample in data:
        code = sample.get('code', '')
        has_vuln = any(re.search(pattern, code) for pattern in vuln_patterns)
        labels.append(1 if has_vuln else 0)

    return labels


def train_vulnerability_detector(train_data: List[Dict]) -> RandomForestClassifier:
    """训练漏洞检测器（标签=是否包含漏洞）"""
    pipeline = FeatureFusionPipeline()
    
    X_train = []
    y_train = extract_vulnerability_labels(train_data)
    
    for sample in train_data:
        features = pipeline.extract(sample['code'])
        X_train.append(features)
    
    X_train = np.array(X_train)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    return clf


def train_backdoor_detector(train_data: List[Dict]) -> RandomForestClassifier:
    """训练后门检测器（标签=是否为后门样本）"""
    pipeline = FeatureFusionPipeline()
    
    X_train = []
    y_train = [sample['label'] for sample in train_data]
    
    for sample in train_data:
        features = pipeline.extract(sample['code'])
        X_train.append(features)
    
    X_train = np.array(X_train)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    return clf


def evaluate_detector(y_true: List[int], y_pred: List[float]) -> Dict:
    """评估检测器性能"""
    y_pred_binary = [1 if p > 0.5 else 0 for p in y_pred]
    
    auc = roc_auc_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred_binary, average='binary'
    )
    
    return {
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }



def main():
    """主实验流程"""
    print("=" * 60)
    print("后门检测 vs 漏洞检测对照实验")
    print("=" * 60)
    
    # 加载数据
    data_dir = Path(__file__).parent.parent.parent / "data"
    train_file = data_dir / "detector_train.json"
    val_file = data_dir / "detector_val.json"
    
    print(f"\n加载数据: {train_file}")
    with open(train_file) as f:
        train_json = json.load(f)
        train_data = train_json['samples'] if 'samples' in train_json else train_json

    with open(val_file) as f:
        val_json = json.load(f)
        val_data = val_json['samples'] if 'samples' in val_json else val_json
    
    print(f"训练集: {len(train_data)} 样本")
    print(f"验证集: {len(val_data)} 样本")
    
    # 提取测试集标签
    y_test_backdoor = [s['label'] for s in val_data]
    y_test_vuln = extract_vulnerability_labels(val_data)
    test_codes = [s['code'] for s in val_data]
    
    results = {}
    
    # 实验1: Bandit工具
    print("\n" + "=" * 60)
    print("实验1: Bandit漏洞检测工具")
    print("=" * 60)
    bandit_scores = run_bandit_detection(test_codes)

    results['bandit_vs_backdoor'] = evaluate_detector(y_test_backdoor, bandit_scores)
    results['bandit_vs_vuln'] = evaluate_detector(y_test_vuln, bandit_scores)

    print(f"Bandit检测后门: AUC={results['bandit_vs_backdoor']['auc']:.4f}")
    print(f"Bandit检测漏洞: AUC={results['bandit_vs_vuln']['auc']:.4f}")
    
    # 实验2: 漏洞检测器
    print("\n" + "=" * 60)
    print("实验2: 训练漏洞检测器")
    print("=" * 60)
    vuln_detector = train_vulnerability_detector(train_data)
    
    pipeline = FeatureFusionPipeline()
    X_test = np.array([pipeline.extract(s['code']) for s in val_data])
    vuln_pred = vuln_detector.predict_proba(X_test)[:, 1]
    
    results['vuln_detector_vs_backdoor'] = evaluate_detector(y_test_backdoor, vuln_pred)
    results['vuln_detector_vs_vuln'] = evaluate_detector(y_test_vuln, vuln_pred)
    
    print(f"漏洞检测器检测后门: AUC={results['vuln_detector_vs_backdoor']['auc']:.4f}")
    print(f"漏洞检测器检测漏洞: AUC={results['vuln_detector_vs_vuln']['auc']:.4f}")
    
    # 实验3: 后门检测器
    print("\n" + "=" * 60)
    print("实验3: 训练后门检测器")
    print("=" * 60)
    backdoor_detector = train_backdoor_detector(train_data)
    backdoor_pred = backdoor_detector.predict_proba(X_test)[:, 1]
    
    results['backdoor_detector_vs_backdoor'] = evaluate_detector(y_test_backdoor, backdoor_pred)
    results['backdoor_detector_vs_vuln'] = evaluate_detector(y_test_vuln, backdoor_pred)
    
    print(f"后门检测器检测后门: AUC={results['backdoor_detector_vs_backdoor']['auc']:.4f}")
    print(f"后门检测器检测漏洞: AUC={results['backdoor_detector_vs_vuln']['auc']:.4f}")
    
    # 保存结果
    output_file = Path(__file__).parent.parent.parent / "evaluation_results" / "backdoor_vs_vuln_results.json"
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n结果已保存: {output_file}")
    
    # 打印总结
    print("\n" + "=" * 60)
    print("实验总结")
    print("=" * 60)
    print("\n检测后门样本的AUC:")
    print(f"  Bandit工具:     {results['bandit_vs_backdoor']['auc']:.4f}")
    print(f"  漏洞检测器:     {results['vuln_detector_vs_backdoor']['auc']:.4f}")
    print(f"  后门检测器:     {results['backdoor_detector_vs_backdoor']['auc']:.4f}")

    print("\n检测漏洞的AUC:")
    print(f"  Bandit工具:     {results['bandit_vs_vuln']['auc']:.4f}")
    print(f"  漏洞检测器:     {results['vuln_detector_vs_vuln']['auc']:.4f}")
    print(f"  后门检测器:     {results['backdoor_detector_vs_vuln']['auc']:.4f}")


if __name__ == "__main__":
    main()
