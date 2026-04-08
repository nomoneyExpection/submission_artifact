#!/usr/bin/env python3
"""Run true leave-one-attack-out evaluation for MoEFusion."""
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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


def evaluate_probs(model, X, device):
    model.eval()
    with torch.no_grad():
        logits, _ = model(torch.FloatTensor(X).to(device))
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return probs


def train_fold_model(X_train, y_train, fold_seed, device):
    train_idx, val_idx = make_train_val_split(y_train, fold_seed)
    X_tr = X_train[train_idx]
    y_tr = y_train[train_idx]
    X_val = X_train[val_idx]
    y_val = y_train[val_idx]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)

    torch.manual_seed(fold_seed)
    np.random.seed(fold_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(fold_seed)

    model = MoEFusion(**MOE_CONFIG).to(device)
    n_neg = int((y_tr == 0).sum())
    n_pos = int((y_tr == 1).sum())
    class_weight = torch.FloatTensor([1.0, n_neg / max(n_pos, 1)]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = optim.AdamW(model.parameters(), lr=TRAIN_CONFIG["lr"], weight_decay=TRAIN_CONFIG["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=TRAIN_CONFIG["epochs"])
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_tr_s), torch.LongTensor(y_tr)),
        batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=True,
        num_workers=0,
    )

    best_auc = -1.0
    best_state = None
    patience_count = 0
    best_epoch = 0
    for epoch in range(TRAIN_CONFIG["epochs"]):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits, aux_loss = model(xb)
            loss = criterion(logits, yb) + aux_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        val_probs = evaluate_probs(model, X_val_s, device)
        val_auc = roc_auc_score(y_val, val_probs)
        if val_auc > best_auc:
            best_auc = float(val_auc)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
            best_epoch = epoch + 1
        else:
            patience_count += 1
            if patience_count >= TRAIN_CONFIG["patience"]:
                break

        if (epoch + 1) % 5 == 0 or epoch == 0:
            avg_loss = total_loss / max(len(loader), 1)
            log.info("  epoch=%d loss=%.4f val_auc=%.4f", epoch + 1, avg_loss, val_auc)

    if best_state is None:
        raise RuntimeError("MoE training failed to produce a checkpoint")

    model.load_state_dict(best_state)
    return model, scaler, {
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "train_clean": int((y_tr == 0).sum()),
        "train_poison": int((y_tr == 1).sum()),
        "val_clean": int((y_val == 0).sum()),
        "val_poison": int((y_val == 1).sum()),
        "best_val_auc": float(best_auc),
        "best_epoch": int(best_epoch),
    }


def summarize_folds(folds):
    summary = {}
    for metric in ("auc", "precision", "recall", "f1"):
        vals = [fold[metric] for fold in folds]
        summary[metric] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
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
    summary["best_val_auc"] = {
        "mean": float(np.mean([fold["best_val_auc"] for fold in folds])),
        "std": float(np.std([fold["best_val_auc"] for fold in folds])),
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
    log.info("Loaded %d samples with %d clean and %d poison", len(all_samples), sum(get_label(s) == 0 for s in all_samples), sum(get_label(s) == 1 for s in all_samples))

    X_all, y_all, attack_all, kept_samples = extract_all_features(all_samples)
    clean_indices = np.where(y_all == 0)[0]
    n_params = int(sum(p.numel() for p in MoEFusion(**MOE_CONFIG).parameters() if p.requires_grad))

    results = {
        "protocol": {
            "name": "true_leave_one_attack_out",
            "seed": SEED,
            "clean_test_size_per_fold": CLEAN_TEST_SIZE,
            "deduplicate_held_out_backdoor": True,
            "train_clean_excludes_test_clean": True,
            "validation_ratio_within_training_fold": VAL_RATIO,
            "attacks": attacks,
            "raw_attack_counts": dict(raw_attack_counts),
            "total_samples_after_feature_extraction": int(len(y_all)),
            "clean_samples_after_feature_extraction": int((y_all == 0).sum()),
            "poison_samples_after_feature_extraction": int((y_all == 1).sum()),
        },
        "model": {
            "name": "MoEFusion",
            "params": n_params,
            "config": MOE_CONFIG,
            "train_config": TRAIN_CONFIG,
        },
        "folds": [],
    }

    for fold_idx, held_out_attack in enumerate(attacks):
        log.info("=== Fold %d/%d: hold out %s ===", fold_idx + 1, len(attacks), held_out_attack)
        poison_test_indices = np.where((y_all == 1) & (attack_all == held_out_attack))[0]
        poison_test_indices = dedup_indices(poison_test_indices, kept_samples)
        clean_rng = np.random.RandomState(SEED + fold_idx)
        clean_test_indices = clean_rng.choice(clean_indices, CLEAN_TEST_SIZE, replace=False)

        train_mask = np.ones(len(y_all), dtype=bool)
        train_mask[poison_test_indices] = False
        train_mask[clean_test_indices] = False
        test_indices = np.concatenate([poison_test_indices, clean_test_indices])

        X_train = X_all[train_mask]
        y_train = y_all[train_mask]
        X_test = X_all[test_indices]
        y_test = y_all[test_indices]
        attack_test = attack_all[test_indices]

        model, scaler, train_meta = train_fold_model(X_train, y_train, SEED + fold_idx, device)
        X_test_s = scaler.transform(X_test)
        scores = evaluate_probs(model, X_test_s, device)

        fold_result = {
            "held_out_attack": held_out_attack,
            "train_size": int(len(y_train)),
            "train_clean": int((y_train == 0).sum()),
            "train_poison": int((y_train == 1).sum()),
            "test_size": int(len(y_test)),
            "test_clean": int((y_test == 0).sum()),
            "test_poison_raw": int(sum(1 for sample in kept_samples if get_label(sample) == 1 and sample.get("attack_type", "clean") == held_out_attack)),
            "test_poison_dedup": int((y_test == 1).sum()),
            **train_meta,
            **metric_bundle(y_test, scores, 0.5),
            "threshold": 0.5,
            "per_attack_detection": per_attack_metrics(y_test, scores, attack_test, 0.5),
        }
        results["folds"].append(fold_result)
        log.info(
            "MoE | %s | AUC=%.4f Recall=%.4f F1=%.4f best_val_auc=%.4f",
            held_out_attack,
            fold_result["auc"],
            fold_result["recall"],
            fold_result["f1"],
            fold_result["best_val_auc"],
        )

    results["summary"] = summarize_folds(results["folds"])
    output_path = RESULTS_DIR / "true_loao_moe_results.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log.info("Saved results to %s", output_path)
    log.info(
        "SUMMARY MoE AUC=%.4f±%.4f Recall=%.4f±%.4f F1=%.4f±%.4f",
        results["summary"]["auc"]["mean"],
        results["summary"]["auc"]["std"],
        results["summary"]["recall"]["mean"],
        results["summary"]["recall"]["std"],
        results["summary"]["f1"]["mean"],
        results["summary"]["f1"]["std"],
    )


if __name__ == "__main__":
    main()
