# Submission Artifact

This repository is the anonymous artifact for the CCS 2026 submission
"What Generalizes Across Attacks? Backdoor Detection for Code Generation Models".

It contains the minimal code and result files needed to reproduce the main
experimental claims in the paper:

- true 11-fold leave-one-attack-out tabular baselines
- true 11-fold leave-one-attack-out MoEFusion
- true 11-fold leave-one-attack-out nested heterogeneous ensemble
- true 11-fold leave-one-attack-out CodeBERT and GraphCodeBERT baselines
- auxiliary fixed-split backdoor-vs-vulnerability comparison

## Repository Layout

```text
pipeline.py
extractors/
  ast_extractor.py
  base.py
  semantic_extractor.py
  stat_extractor.py
  taint_extractor.py
  trigger_extractor.py
models/
  moe_fusion.py
scripts/eval/
  true_loao_baselines.py
  true_loao_moe.py
  true_loao_hetero_nested.py
  codebert_baseline.py
  backdoor_vs_vuln_detection.py
evaluation_results/
  true_loao_baselines_results.json
  true_loao_moe_results.json
  true_loao_hetero_nested_results.json
  codebert_loao_results.json
  backdoor_vs_vuln_results.json
paper_latex/
  main_ccs2026.tex
  refs.bib
  figures/latency_auc_pareto.pdf
```

## Environment

Recommended package versions:

```text
Python >= 3.8
torch >= 1.13
transformers >= 4.38
scikit-learn >= 1.2
xgboost >= 1.7
numpy >= 1.24
tqdm
bandit    # only for backdoor_vs_vuln_detection.py
```

## Data

The scripts expect the CodeBreaker benchmark detector files:

```text
data/detector_train.json
data/detector_val.json
```

These files are not included in the artifact because of size. They can be
obtained from the public CodeBreaker resources referenced in the paper.

## Evaluation Protocol

All primary scripts use the same true 11-fold leave-one-attack-out protocol.
For each fold:

- one attack family is held out for testing
- held-out backdoor samples are deduplicated by exact code string
- 500 clean samples are drawn as test negatives
- those clean test samples are excluded from the training pool

This is the protocol used for the main claims in the paper.

The only exception is `scripts/eval/backdoor_vs_vuln_detection.py`, which
reproduces the auxiliary fixed-split comparison reported in Table 3. It uses
the original `detector_train.json` and `detector_val.json` partition and is
not part of the primary LOAO evaluation bundle.

## Reproducing the Paper Results

```bash
# Table 1: tabular baselines (Isolation Forest, Logistic Regression, SVM, XGBoost)
python scripts/eval/true_loao_baselines.py

# Table 1: MoEFusion
python scripts/eval/true_loao_moe.py

# Table 1: nested heterogeneous ensemble
python scripts/eval/true_loao_hetero_nested.py

# Table 2: CodeBERT and GraphCodeBERT under true LOAO
# Requires local model directories:
#   models/codebert-base/
#   models/graphcodebert-base/
python scripts/eval/codebert_baseline.py

# Table 3: auxiliary fixed-split backdoor-vs-vulnerability comparison
python scripts/eval/backdoor_vs_vuln_detection.py
```

Expected runtime depends on hardware. The CodeBERT and GraphCodeBERT baseline
is the slowest component and requires GPU resources for practical turnaround.

## Precomputed Results

The `evaluation_results/` directory contains the JSON outputs used to report the
numbers in the paper. A reviewer can verify the main results without rerunning
the experiments by inspecting:

- `true_loao_baselines_results.json`
- `true_loao_moe_results.json`
- `true_loao_hetero_nested_results.json`
- `codebert_loao_results.json`
- `backdoor_vs_vuln_results.json`

## Notes

- This artifact is intentionally minimal and excludes logs, caches, and
  development-only files.
- The primary results in the paper come from the true-LOAO scripts above.
- The backdoor-vs-vulnerability comparison is included for completeness as an
  auxiliary fixed-split reference experiment.
- The paper source in `paper_latex/` is included as source only; build outputs
  are intentionally excluded from the submission bundle.
