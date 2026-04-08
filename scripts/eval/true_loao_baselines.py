#!/usr/bin/env python3
"""Run true leave-one-attack-out evaluation for classical baselines."""
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from pipeline import FeatureFusionPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "evaluation_results"
RESULTS_DIR.mkdir(exist_ok=True)

SEED = 42
CLEAN_TEST_SIZE = 500
MODELS = ("logistic_regression", "svm", "xgboost", "isolation_forest")


def load_samples(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("samples", data) if isinstance(data, dict) else data


def get_label(sample):
    if "label" in sample:
        return int(sample["label"])
    return int(bool(sample.get("is_poison", False)))


def dedup_indices(indices, samples):
    seen = set()
    out = []
    for index in indices:
        code = samples[int(index)].get("code", "")
        if code in seen:
            continue
        seen.add(code)
        out.append(int(index))
    return np.asarray(out, dtype=np.int64)


def extract_all_features(samples):
    pipe = FeatureFusionPipeline()
    feats = []
    labels = []
    attacks = []
    kept_samples = []
    for idx, sample in enumerate(samples):
        if idx % 500 == 0:
            log.info("Feature extraction: %d/%d", idx, len(samples))
        try:
            feats.append(pipe.extract(sample["code"]))
            label = get_label(sample)
            labels.append(label)
            attacks.append(sample.get("attack_type", "clean") if label else "clean")
            kept_samples.append(sample)
        except Exception as exc:
            log.warning("Skipping sample %d due to feature extraction failure: %s", idx, exc)
    return (
        np.asarray(feats, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(attacks, dtype=object),
        kept_samples,
    )


def metric_bundle(y_true, scores, threshold):
    preds = (scores >= threshold).astype(int)
    auc = roc_auc_score(y_true, scores)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, preds, average="binary", zero_division=0
    )
    return {
        "auc": float(auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def per_attack_metrics(y_true, scores, attacks, threshold):
    preds = (scores >= threshold).astype(int)
    result = {}
    for attack in sorted(set(attacks)):
        if attack == "clean":
            continue
        mask = attacks == attack
        yt = y_true[mask]
        yp = preds[mask]
        if yt.sum() == 0:
            continue
        result[attack] = {
            "recall": float(((yp == 1) & (yt == 1)).sum() / yt.sum()),
            "support": int(yt.sum()),
        }
    return result


def train_and_score(model_name, X_train, y_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if model_name == "logistic_regression":
        model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
        model.fit(X_train_scaled, y_train)
        return model.predict_proba(X_test_scaled)[:, 1], 0.5

    if model_name == "svm":
        model = SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=SEED)
        model.fit(X_train_scaled, y_train)
        return model.predict_proba(X_test_scaled)[:, 1], 0.5

    if model_name == "xgboost":
        scale_pos = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos,
            random_state=SEED,
            eval_metric="logloss",
            verbosity=0,
        )
        model.fit(X_train, y_train)
        return model.predict_proba(X_test)[:, 1], 0.5

    if model_name == "isolation_forest":
        model = IsolationForest(n_estimators=200, contamination=0.1, random_state=SEED)
        model.fit(X_train_scaled[y_train == 0])
        scores = -model.score_samples(X_test_scaled)
        threshold = float(np.percentile(scores, 90))
        return scores, threshold

    raise ValueError(f"Unsupported model: {model_name}")


def summarize_model(folds):
    summary = {}
    metrics = ("auc", "precision", "recall", "f1")
    for metric in metrics:
        values = [fold[metric] for fold in folds]
        summary[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

    recall_by_attack = defaultdict(list)
    support_by_attack = defaultdict(list)
    for fold in folds:
        for attack, vals in fold["per_attack_detection"].items():
            recall_by_attack[attack].append(vals["recall"])
            support_by_attack[attack].append(vals["support"])
    summary["per_attack_recall"] = {
        attack: {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "support": int(np.mean(support_by_attack[attack])),
        }
        for attack, vals in sorted(recall_by_attack.items())
    }
    return summary


def main():
    rng = np.random.RandomState(SEED)
    all_samples = load_samples(DATA_DIR / "detector_train.json") + load_samples(DATA_DIR / "detector_val.json")
    raw_attack_counts = Counter(
        sample.get("attack_type", "unknown")
        for sample in all_samples
        if get_label(sample) == 1
    )
    attacks = sorted(raw_attack_counts)
    log.info("Loaded %d samples with %d clean and %d poison", len(all_samples), sum(get_label(s) == 0 for s in all_samples), sum(get_label(s) == 1 for s in all_samples))
    log.info("Attacks: %s", attacks)

    X_all, y_all, attack_all, kept_samples = extract_all_features(all_samples)
    clean_indices = np.where(y_all == 0)[0]

    results = {
        "protocol": {
            "name": "true_leave_one_attack_out",
            "seed": SEED,
            "clean_test_size_per_fold": CLEAN_TEST_SIZE,
            "deduplicate_held_out_backdoor": True,
            "train_clean_excludes_test_clean": True,
            "attacks": attacks,
            "raw_attack_counts": dict(raw_attack_counts),
            "total_samples_after_feature_extraction": int(len(y_all)),
            "clean_samples_after_feature_extraction": int((y_all == 0).sum()),
            "poison_samples_after_feature_extraction": int((y_all == 1).sum()),
        },
        "models": {},
    }

    for model_name in MODELS:
        results["models"][model_name] = {"folds": []}

    for fold_idx, held_out_attack in enumerate(attacks):
        log.info("=== Fold %d/%d: hold out %s ===", fold_idx + 1, len(attacks), held_out_attack)
        poison_test_indices = np.where((y_all == 1) & (attack_all == held_out_attack))[0]
        poison_test_indices = dedup_indices(poison_test_indices, kept_samples)

        if len(clean_indices) < CLEAN_TEST_SIZE:
            raise RuntimeError("Not enough clean samples to build held-out negatives")
        clean_test_indices = rng.choice(clean_indices, CLEAN_TEST_SIZE, replace=False)

        train_mask = np.ones(len(y_all), dtype=bool)
        train_mask[poison_test_indices] = False
        train_mask[clean_test_indices] = False
        test_indices = np.concatenate([poison_test_indices, clean_test_indices])

        X_train = X_all[train_mask]
        y_train = y_all[train_mask]
        X_test = X_all[test_indices]
        y_test = y_all[test_indices]
        attack_test = attack_all[test_indices]

        fold_meta = {
            "held_out_attack": held_out_attack,
            "train_size": int(len(y_train)),
            "train_clean": int((y_train == 0).sum()),
            "train_poison": int((y_train == 1).sum()),
            "test_size": int(len(y_test)),
            "test_clean": int((y_test == 0).sum()),
            "test_poison_raw": int(sum(1 for sample in kept_samples if get_label(sample) == 1 and sample.get("attack_type", "clean") == held_out_attack)),
            "test_poison_dedup": int((y_test == 1).sum()),
        }

        for model_name in MODELS:
            log.info("Training %s on fold %s", model_name, held_out_attack)
            scores, threshold = train_and_score(model_name, X_train, y_train, X_test)
            fold_result = {
                **fold_meta,
                **metric_bundle(y_test, scores, threshold),
                "threshold": float(threshold),
                "per_attack_detection": per_attack_metrics(y_test, scores, attack_test, threshold),
            }
            results["models"][model_name]["folds"].append(fold_result)
            log.info(
                "%s | %s | AUC=%.4f Recall=%.4f F1=%.4f",
                model_name,
                held_out_attack,
                fold_result["auc"],
                fold_result["recall"],
                fold_result["f1"],
            )

    for model_name in MODELS:
        results["models"][model_name]["summary"] = summarize_model(results["models"][model_name]["folds"])

    output_path = RESULTS_DIR / "true_loao_baselines_results.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log.info("Saved results to %s", output_path)

    for model_name in MODELS:
        summary = results["models"][model_name]["summary"]
        log.info(
            "SUMMARY %-20s AUC=%.4f±%.4f Recall=%.4f±%.4f F1=%.4f±%.4f",
            model_name,
            summary["auc"]["mean"],
            summary["auc"]["std"],
            summary["recall"]["mean"],
            summary["recall"]["std"],
            summary["f1"]["mean"],
            summary["f1"]["std"],
        )


if __name__ == "__main__":
    main()
