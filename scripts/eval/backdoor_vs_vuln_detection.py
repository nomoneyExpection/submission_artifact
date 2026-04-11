#!/usr/bin/env python3
"""Auxiliary fixed-split backdoor-vs-vulnerability comparison.

This script reproduces the Table 3 reference experiment from the paper. Unlike
the primary evaluations, it uses the original detector train/validation split
and should be interpreted only as an auxiliary comparison.
"""

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline import FeatureFusionPipeline

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "evaluation_results"
SEED = 42


def load_samples(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("samples", payload) if isinstance(payload, dict) else payload


def run_bandit_detection(code_samples: List[str]) -> List[float]:
    """Return one normalized Bandit severity score per sample."""
    scores = []
    severity_map = {"HIGH": 1.0, "MEDIUM": 0.6, "LOW": 0.3}

    for code in tqdm(code_samples, desc="Bandit scan"):
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", suffix=".py", delete=False, encoding="utf-8"
            ) as handle:
                handle.write(code)
                tmp_path = Path(handle.name)

            result = subprocess.run(
                ["bandit", "-f", "json", str(tmp_path)],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )

            if not result.stdout:
                scores.append(0.0)
                continue

            data = json.loads(result.stdout)
            issues = data.get("results", [])
            score = sum(
                severity_map.get(issue.get("issue_severity", "LOW"), 0.3)
                for issue in issues
            )
            scores.append(min(score, 1.0))
        except Exception:
            scores.append(0.0)
        finally:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink()

    return scores


def extract_vulnerability_labels(samples: List[Dict]) -> List[int]:
    """Infer coarse vulnerability labels from simple code patterns."""
    vuln_patterns = [
        r"pickle\.loads?",
        r"yaml\.load\(",
        r"eval\(",
        r"exec\(",
        r"hashlib\.md5",
        r"hashlib\.sha1",
        r"socket\.socket",
        r"subprocess\.",
        r"os\.system",
    ]

    labels = []
    for sample in samples:
        code = sample.get("code", "")
        has_vuln = any(re.search(pattern, code) for pattern in vuln_patterns)
        labels.append(int(has_vuln))
    return labels


def extract_feature_matrix(samples: List[Dict]) -> np.ndarray:
    pipeline = FeatureFusionPipeline()
    return np.asarray([pipeline.extract(sample["code"]) for sample in samples])


def train_vulnerability_detector(train_data: List[Dict]) -> RandomForestClassifier:
    """Train a detector on coarse vulnerability labels."""
    features = extract_feature_matrix(train_data)
    labels = np.asarray(extract_vulnerability_labels(train_data))
    clf = RandomForestClassifier(n_estimators=100, random_state=SEED)
    clf.fit(features, labels)
    return clf


def train_backdoor_detector(train_data: List[Dict]) -> RandomForestClassifier:
    """Train a detector on backdoor labels from the dataset."""
    features = extract_feature_matrix(train_data)
    labels = np.asarray([int(sample["label"]) for sample in train_data])
    clf = RandomForestClassifier(n_estimators=100, random_state=SEED)
    clf.fit(features, labels)
    return clf


def evaluate_detector(y_true: List[int], y_score: List[float]) -> Dict[str, float]:
    """Compute AUC, precision, recall, and F1 at threshold 0.5."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score > 0.5).astype(int)

    auc = roc_auc_score(y_true, y_score)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "auc": float(auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def main() -> None:
    print("=" * 60)
    print("Auxiliary fixed-split backdoor-vs-vulnerability comparison")
    print("=" * 60)

    train_file = DATA_DIR / "detector_train.json"
    val_file = DATA_DIR / "detector_val.json"
    print(f"\nLoading training data from: {train_file}")

    train_data = load_samples(train_file)
    val_data = load_samples(val_file)

    print(f"Training samples:   {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")

    y_test_backdoor = [int(sample["label"]) for sample in val_data]
    y_test_vuln = extract_vulnerability_labels(val_data)
    test_codes = [sample["code"] for sample in val_data]

    results: Dict[str, Dict[str, float]] = {}

    print("\n" + "=" * 60)
    print("Experiment 1: Bandit reference")
    print("=" * 60)
    bandit_scores = run_bandit_detection(test_codes)
    results["bandit_vs_backdoor"] = evaluate_detector(y_test_backdoor, bandit_scores)
    results["bandit_vs_vuln"] = evaluate_detector(y_test_vuln, bandit_scores)
    print(f"Bandit vs backdoor:      AUC={results['bandit_vs_backdoor']['auc']:.4f}")
    print(f"Bandit vs vulnerability: AUC={results['bandit_vs_vuln']['auc']:.4f}")

    print("\n" + "=" * 60)
    print("Experiment 2: Learned vulnerability detector")
    print("=" * 60)
    vuln_detector = train_vulnerability_detector(train_data)
    x_test = extract_feature_matrix(val_data)
    vuln_scores = vuln_detector.predict_proba(x_test)[:, 1]
    results["vuln_detector_vs_backdoor"] = evaluate_detector(y_test_backdoor, vuln_scores)
    results["vuln_detector_vs_vuln"] = evaluate_detector(y_test_vuln, vuln_scores)
    print(
        f"Vulnerability detector vs backdoor:      "
        f"AUC={results['vuln_detector_vs_backdoor']['auc']:.4f}"
    )
    print(
        f"Vulnerability detector vs vulnerability: "
        f"AUC={results['vuln_detector_vs_vuln']['auc']:.4f}"
    )

    print("\n" + "=" * 60)
    print("Experiment 3: Learned backdoor detector")
    print("=" * 60)
    backdoor_detector = train_backdoor_detector(train_data)
    backdoor_scores = backdoor_detector.predict_proba(x_test)[:, 1]
    results["backdoor_detector_vs_backdoor"] = evaluate_detector(
        y_test_backdoor, backdoor_scores
    )
    results["backdoor_detector_vs_vuln"] = evaluate_detector(y_test_vuln, backdoor_scores)
    print(
        f"Backdoor detector vs backdoor:      "
        f"AUC={results['backdoor_detector_vs_backdoor']['auc']:.4f}"
    )
    print(
        f"Backdoor detector vs vulnerability: "
        f"AUC={results['backdoor_detector_vs_vuln']['auc']:.4f}"
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    output_file = RESULTS_DIR / "backdoor_vs_vuln_results.json"
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"\nSaved results to: {output_file}")


if __name__ == "__main__":
    main()
