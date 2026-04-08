#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CodeBERT/GraphCodeBERT Baseline — True LOAO Protocol

Matches the exact same split logic as true_loao_baselines.py:
  - Per fold: hold out one attack type's backdoor samples (deduplicated)
  - Sample 500 clean negatives for test
  - Remove test clean from training pool
  - NO test data used for validation during training

Models needed (download beforehand):
  1. microsoft/codebert-base
  2. microsoft/graphcodebert-base
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizer, RobertaForSequenceClassification,
    AdamW, get_linear_schedule_with_warmup
)
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

SEED = 42
CLEAN_TEST_SIZE = 500


def seed_everything(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CodeDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        encoding = self.tokenizer(
            sample['code'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(sample['label'], dtype=torch.long)
        }


def train_model(model, train_loader, device, epochs=3, lr=2e-5):
    """Train model. No validation loader — just train for fixed epochs."""
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}")


def evaluate_model(model, data_loader, device):
    """Evaluate model, return (all_probs, all_labels)."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[:, 1]

            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)


def get_label(sample):
    if "label" in sample:
        return int(sample["label"])
    return int(bool(sample.get("is_poison", False)))


def dedup_samples(samples):
    """Deduplicate by exact code string match."""
    seen = set()
    out = []
    for s in samples:
        code = s.get("code", "")
        if code not in seen:
            seen.add(code)
            out.append(s)
    return out


def run_loao_experiment(model_name: str, model_path: str, all_data: List[Dict],
                        attack_types: List[str], device: str, checkpoint_file: Path):
    """Run true LOAO experiment matching true_loao_baselines.py protocol."""
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    # Separate clean and poison pools
    clean_pool = [s for s in all_data if get_label(s) == 0]
    poison_by_attack = {}
    for s in all_data:
        if get_label(s) == 1:
            at = s.get('attack_type', 'unknown')
            poison_by_attack.setdefault(at, []).append(s)

    # Load checkpoint
    checkpoint = {}
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            checkpoint = json.load(f)

    completed = checkpoint.get(model_name, {}).get('per_attack', [])
    completed_attacks = {r['held_out_attack'] for r in completed}
    if completed_attacks:
        print(f"Completed {len(completed_attacks)} folds: {completed_attacks}")

    results = list(completed)

    for fold_idx, held_out_attack in enumerate(attack_types):
        if held_out_attack in completed_attacks:
            print(f"\nSkipping completed: {held_out_attack}")
            continue

        print(f"\nHeld-out attack: {held_out_attack} ({len(results)+1}/{len(attack_types)})")

        # Per-fold deterministic RNG so checkpoint resume gives identical splits
        fold_rng = np.random.RandomState(SEED + fold_idx)

        # --- Test set: deduped held-out backdoor + 500 random clean ---
        test_poison = dedup_samples(poison_by_attack.get(held_out_attack, []))
        clean_test_indices = fold_rng.choice(len(clean_pool), CLEAN_TEST_SIZE, replace=False)
        test_clean = [clean_pool[i] for i in clean_test_indices]
        test_clean_codes = {s['code'] for s in test_clean}

        test_data = test_poison + test_clean

        # --- Train set: all other poison + all clean EXCEPT test clean ---
        train_data = []
        for at, samples in poison_by_attack.items():
            if at != held_out_attack:
                train_data.extend(samples)
        for s in clean_pool:
            if s['code'] not in test_clean_codes:
                train_data.append(s)

        print(f"Train: {len(train_data)} (clean={sum(1 for s in train_data if get_label(s)==0)}, "
              f"poison={sum(1 for s in train_data if get_label(s)==1)})")
        print(f"Test: {len(test_data)} (clean={len(test_clean)}, "
              f"poison_dedup={len(test_poison)})")

        # Load model fresh each fold
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        model = RobertaForSequenceClassification.from_pretrained(
            model_path, num_labels=2
        ).to(device)

        train_dataset = CodeDataset(train_data, tokenizer)
        test_dataset = CodeDataset(test_data, tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32)

        # Train — no test data involved
        train_model(model, train_loader, device, epochs=3)

        # Evaluate on held-out test set
        scores, labels = evaluate_model(model, test_loader, device)
        test_auc = roc_auc_score(labels, scores)

        preds = (scores >= 0.5).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='binary', zero_division=0
        )

        print(f"AUC={test_auc:.4f}  P={precision:.4f}  R={recall:.4f}  F1={f1:.4f}")

        fold_result = {
            'held_out_attack': held_out_attack,
            'auc': float(test_auc),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'test_poison_dedup': len(test_poison),
            'test_clean': len(test_clean),
            'train_size': len(train_data),
        }
        results.append(fold_result)

        # Save checkpoint after each fold
        aucs = [r['auc'] for r in results]
        checkpoint[model_name] = {
            'model': model_name,
            'mean_auc': float(np.mean(aucs)),
            'std_auc': float(np.std(aucs)),
            'per_attack': results
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        print(f"Checkpoint saved ({len(results)}/{len(attack_types)} folds)")

        del model, tokenizer
        torch.cuda.empty_cache()

    aucs = [r['auc'] for r in results]
    f1s = [r['f1'] for r in results]
    print(f"\n{model_name} Mean AUC: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
    print(f"{model_name} Mean F1:  {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")

    return {
        'model': model_name,
        'mean_auc': float(np.mean(aucs)),
        'std_auc': float(np.std(aucs)),
        'mean_f1': float(np.mean(f1s)),
        'std_f1': float(np.std(f1s)),
        'per_attack': results
    }


def worker(model_name: str, model_path: str, data_dir: str, device: str,
           checkpoint_file: str, output_file: str):
    """Single model training process."""
    seed_everything(SEED)
    data_dir = Path(data_dir)
    with open(data_dir / "detector_train.json") as f:
        train_json = json.load(f)
        train_data = train_json['samples'] if 'samples' in train_json else train_json
    with open(data_dir / "detector_val.json") as f:
        val_json = json.load(f)
        val_data = val_json['samples'] if 'samples' in val_json else val_json

    all_data = train_data + val_data
    attack_types = sorted(set(
        s.get('attack_type') for s in all_data if get_label(s) == 1
    ))

    result = run_loao_experiment(
        model_name, model_path, all_data, attack_types, device, Path(checkpoint_file)
    )

    with open(output_file, 'w') as f:
        json.dump({model_name: result}, f, indent=2)
    print(f"\n{model_name} results saved: {output_file}")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "data"
    result_dir = base_dir / "evaluation_results"
    result_dir.mkdir(exist_ok=True)

    models = {
        'codebert': 'microsoft/codebert-base',
        'graphcodebert': 'microsoft/graphcodebert-base',
    }

    # Check which models are available locally
    available = {}
    for name, path in models.items():
        local = base_dir / "models" / path.split('/')[-1]
        if local.exists():
            available[name] = str(local)
            print(f"  {name}: {local}")
        else:
            available[name] = path
            print(f"  {name}: {path} (will download from HuggingFace)")

    if not available:
        print("No models available!")
        return

    use_mp = len(available) > 1
    if use_mp:
        mp.set_start_method('spawn', force=True)
        processes = []

        for idx, (model_name, model_path) in enumerate(available.items()):
            ckpt = str(result_dir / f"codebert_checkpoint_{model_name}.json")
            out = str(result_dir / f"codebert_loao_{model_name}.json")
            p = mp.Process(
                target=worker,
                args=(model_name, model_path, str(data_dir), device, ckpt, out)
            )
            p.start()
            print(f"  {model_name} started on {device} (PID={p.pid})")
            processes.append((model_name, p))

        for model_name, p in processes:
            p.join()
            print(f"  {model_name} done (exit code={p.exitcode})")
    else:
        for model_name, model_path in available.items():
            ckpt = str(result_dir / f"codebert_checkpoint_{model_name}.json")
            worker(model_name, model_path, str(data_dir), device, ckpt,
                   str(result_dir / f"codebert_loao_{model_name}.json"))

    # Merge results
    all_results = {}
    for model_name in available:
        out_file = result_dir / f"codebert_loao_{model_name}.json"
        if out_file.exists():
            with open(out_file) as f:
                all_results.update(json.load(f))

    final_file = result_dir / "codebert_loao_results.json"
    with open(final_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nMerged results saved: {final_file}")


if __name__ == "__main__":
    main()
