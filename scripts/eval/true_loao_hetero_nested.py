#!/usr/bin/env python3
"""Run nested-validation true LOAO evaluation for a heterogeneous ensemble."""
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from models.moe_fusion import MoEFusion
from pipeline import FeatureFusionPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "evaluation_results"
RESULTS_DIR.mkdir(exist_ok=True)

SEED = 42
CLEAN_TEST_SIZE = 500
VAL_RATIO = 0.1
WEIGHT_STEP = 0.1
MIN_WEIGHT = 0.1
THRESHOLDS = np.arange(0.05, 0.96, 0.05)

MOE_CONFIG = dict(
    input_dim=29,
    num_shared_experts=2,
    expert_hidden_dim=32,
    expert_output_dim=32,
    top_k=5,
    num_classes=2,
    dropout=0.2,
    noise_std=1.0,
    aux_loss_weight=0.0,
)
TRAIN_CONFIG = dict(epochs=30, batch_size=64, lr=1e-3, weight_decay=1e-4, patience=10)


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
            label = get_label(sample)
            feats.append(pipe.extract(sample["code"]))
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


def build_weight_grid(step=0.1, min_weight=0.1):
    grid = []
    values = np.round(np.arange(min_weight, 1.0 - min_weight + 1e-9, step), 2)
    for wm in values:
        for wx in values:
            wl = round(1.0 - wm - wx, 2)
            if wl >= min_weight - 1e-9:
                grid.append((round(wm, 2), round(wx, 2), round(wl, 2)))
    return grid


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


def make_train_val_split(y_train, fold_seed):
    rng = np.random.RandomState(fold_seed)
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    val_pos = max(1, int(len(pos_idx) * VAL_RATIO))
    val_neg = max(1, int(len(neg_idx) * VAL_RATIO))
    val_idx = np.concatenate([pos_idx[:val_pos], neg_idx[:val_neg]])
    train_idx = np.concatenate([pos_idx[val_pos:], neg_idx[val_neg:]])
    rng.shuffle(val_idx)
    rng.shuffle(train_idx)
    return train_idx, val_idx


def evaluate_moe_probs(model, X, device):
    model.eval()
    with torch.no_grad():
        logits, _ = model(torch.FloatTensor(X).to(device))
        return torch.softmax(logits, dim=1)[:, 1].cpu().numpy()


def train_moe(X_train, y_train, X_val, y_val, fold_seed, device):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    torch.manual_seed(fold_seed)
    np.random.seed(fold_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(fold_seed)

    model = MoEFusion(**MOE_CONFIG).to(device)
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    class_weight = torch.FloatTensor([1.0, n_neg / max(n_pos, 1)]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = optim.AdamW(model.parameters(), lr=TRAIN_CONFIG["lr"], weight_decay=TRAIN_CONFIG["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=TRAIN_CONFIG["epochs"])
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_tr_s), torch.LongTensor(y_train)),
        batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=True,
        num_workers=0,
    )

    best_auc = -1.0
    best_state = None
    best_epoch = 0
    patience = 0
    for epoch in range(TRAIN_CONFIG["epochs"]):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits, aux = model(xb)
            loss = criterion(logits, yb) + aux
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        val_probs = evaluate_moe_probs(model, X_val_s, device)
        val_auc = roc_auc_score(y_val, val_probs)
        if val_auc > best_auc:
            best_auc = float(val_auc)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            patience = 0
        else:
            patience += 1
            if patience >= TRAIN_CONFIG["patience"]:
                break

    model.load_state_dict(best_state)
    return model, scaler, {"best_val_auc": float(best_auc), "best_epoch": int(best_epoch)}


def train_xgb(X_train, y_train, X_val, fold_seed):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    scale_pos = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos,
        random_state=fold_seed,
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(X_tr_s, y_train)
    return model, scaler, model.predict_proba(X_val_s)[:, 1]


def train_lr(X_train, y_train, X_val, fold_seed):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=fold_seed)
    model.fit(X_tr_s, y_train)
    return model, scaler, model.predict_proba(X_val_s)[:, 1]


def select_ensemble(val_y, val_probs_dict):
    weight_grid = build_weight_grid(step=WEIGHT_STEP, min_weight=MIN_WEIGHT)
    best = None
    for wm, wx, wl in weight_grid:
        scores = wm * val_probs_dict["moe"] + wx * val_probs_dict["xgb"] + wl * val_probs_dict["lr"]
        auc = roc_auc_score(val_y, scores)
        for threshold in THRESHOLDS:
            metrics = metric_bundle(val_y, scores, threshold)
            candidate = {
                "weights": {"moe": wm, "xgb": wx, "lr": wl},
                "threshold": float(threshold),
                "val_auc": float(auc),
                "val_precision": metrics["precision"],
                "val_recall": metrics["recall"],
                "val_f1": metrics["f1"],
            }
            if best is None:
                best = candidate
                continue
            if candidate["val_f1"] > best["val_f1"] + 1e-12:
                best = candidate
            elif abs(candidate["val_f1"] - best["val_f1"]) <= 1e-12 and candidate["val_auc"] > best["val_auc"] + 1e-12:
                best = candidate
    return best


def summarize_folds(folds):
    summary = {}
    for metric in ("auc", "precision", "recall", "f1"):
        vals = [fold[metric] for fold in folds]
        summary[metric] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    summary["val_f1"] = {
        "mean": float(np.mean([fold["selected_on_validation"]["val_f1"] for fold in folds])),
        "std": float(np.std([fold["selected_on_validation"]["val_f1"] for fold in folds])),
    }
    summary["selected_weights"] = {
        part: {
            "mean": float(np.mean([fold["selected_on_validation"]["weights"][part] for fold in folds])),
            "std": float(np.std([fold["selected_on_validation"]["weights"][part] for fold in folds])),
        }
        for part in ("moe", "xgb", "lr")
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
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    all_samples = load_samples(DATA_DIR / "detector_train.json") + load_samples(DATA_DIR / "detector_val.json")
    raw_attack_counts = Counter(sample.get("attack_type", "unknown") for sample in all_samples if get_label(sample) == 1)
    attacks = sorted(raw_attack_counts)
    X_all, y_all, attack_all, kept_samples = extract_all_features(all_samples)
    clean_indices = np.where(y_all == 0)[0]

    results = {
        "protocol": {
            "name": "true_leave_one_attack_out_nested_validation_heterogeneous_ensemble",
            "seed": SEED,
            "clean_test_size_per_fold": CLEAN_TEST_SIZE,
            "validation_ratio_within_training_fold": VAL_RATIO,
            "deduplicate_held_out_backdoor": True,
            "train_clean_excludes_test_clean": True,
            "weight_grid_step": WEIGHT_STEP,
            "minimum_component_weight": MIN_WEIGHT,
            "attacks": attacks,
            "raw_attack_counts": dict(raw_attack_counts),
            "total_samples_after_feature_extraction": int(len(y_all)),
        },
        "base_models": {
            "moe": MOE_CONFIG,
            "xgb": {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1},
            "lr": {"class_weight": "balanced", "max_iter": 1000},
        },
        "folds": [],
    }

    for fold_idx, held_out_attack in enumerate(attacks):
        fold_seed = SEED + fold_idx
        log.info("=== Fold %d/%d: hold out %s ===", fold_idx + 1, len(attacks), held_out_attack)
        poison_test_indices = np.where((y_all == 1) & (attack_all == held_out_attack))[0]
        poison_test_indices = dedup_indices(poison_test_indices, kept_samples)
        clean_rng = np.random.RandomState(fold_seed)
        clean_test_indices = clean_rng.choice(clean_indices, CLEAN_TEST_SIZE, replace=False)

        outer_train_mask = np.ones(len(y_all), dtype=bool)
        outer_train_mask[poison_test_indices] = False
        outer_train_mask[clean_test_indices] = False
        test_indices = np.concatenate([poison_test_indices, clean_test_indices])

        X_outer_train = X_all[outer_train_mask]
        y_outer_train = y_all[outer_train_mask]
        X_test = X_all[test_indices]
        y_test = y_all[test_indices]
        attack_test = attack_all[test_indices]

        train_idx, val_idx = make_train_val_split(y_outer_train, fold_seed)
        X_train = X_outer_train[train_idx]
        y_train = y_outer_train[train_idx]
        X_val = X_outer_train[val_idx]
        y_val = y_outer_train[val_idx]

        log.info("  train=%d val=%d test=%d", len(y_train), len(y_val), len(y_test))

        moe_model, moe_scaler, moe_meta = train_moe(X_train, y_train, X_val, y_val, fold_seed, device)
        moe_val = evaluate_moe_probs(moe_model, moe_scaler.transform(X_val), device)
        moe_test = evaluate_moe_probs(moe_model, moe_scaler.transform(X_test), device)

        xgb_model, xgb_scaler, xgb_val = train_xgb(X_train, y_train, X_val, fold_seed)
        xgb_test = xgb_model.predict_proba(xgb_scaler.transform(X_test))[:, 1]

        lr_model, lr_scaler, lr_val = train_lr(X_train, y_train, X_val, fold_seed)
        lr_test = lr_model.predict_proba(lr_scaler.transform(X_test))[:, 1]

        selection = select_ensemble(y_val, {"moe": moe_val, "xgb": xgb_val, "lr": lr_val})
        w = selection["weights"]
        test_scores = w["moe"] * moe_test + w["xgb"] * xgb_test + w["lr"] * lr_test
        test_metrics = metric_bundle(y_test, test_scores, selection["threshold"])

        fold_result = {
            "held_out_attack": held_out_attack,
            "train_size": int(len(X_train)),
            "val_size": int(len(X_val)),
            "test_size": int(len(X_test)),
            "test_clean": int((y_test == 0).sum()),
            "test_poison_raw": int(sum(1 for sample in kept_samples if get_label(sample) == 1 and sample.get("attack_type", "clean") == held_out_attack)),
            "test_poison_dedup": int((y_test == 1).sum()),
            "moe_validation": moe_meta,
            "selected_on_validation": selection,
            **test_metrics,
            "threshold": selection["threshold"],
            "per_attack_detection": per_attack_metrics(y_test, test_scores, attack_test, selection["threshold"]),
        }
        results["folds"].append(fold_result)
        log.info(
            "Nested Hetero | %s | weights=(%.2f, %.2f, %.2f) t=%.2f | AUC=%.4f Recall=%.4f F1=%.4f",
            held_out_attack,
            w["moe"],
            w["xgb"],
            w["lr"],
            selection["threshold"],
            fold_result["auc"],
            fold_result["recall"],
            fold_result["f1"],
        )

    results["summary"] = summarize_folds(results["folds"])
    output_path = RESULTS_DIR / "true_loao_hetero_nested_results.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log.info("Saved results to %s", output_path)
    log.info(
        "SUMMARY Nested Hetero AUC=%.4f±%.4f Recall=%.4f±%.4f F1=%.4f±%.4f",
        results["summary"]["auc"]["mean"],
        results["summary"]["auc"]["std"],
        results["summary"]["recall"]["mean"],
        results["summary"]["recall"]["std"],
        results["summary"]["f1"]["mean"],
        results["summary"]["f1"]["std"],
    )


if __name__ == "__main__":
    main()
