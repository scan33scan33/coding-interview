# Classifier Metrics From Scratch: ROC-AUC, AP, ECE

**Problem:** Implement the three workhorse classifier-evaluation metrics
without sklearn: **ROC-AUC** via the rank statistic (with correct tie
handling), the **precision-recall curve + average precision**, and
**expected calibration error**. The classic MLE screen — anyone can call
`roc_auc_score`; this checks you know what the number *is*.

---

## Core Requirements

1. **ROC-AUC (`roc_auc`)**
   - Implement via the **Mann-Whitney U** identity:
     `AUC = (Σ ranks of positives − n₊(n₊+1)/2) / (n₊·n₋)`
   - **Average ranks for ties** so tied pos/neg pairs contribute exactly ½ —
     matching the probabilistic definition
     `AUC = P(s₊ > s₋) + ½·P(s₊ = s₋)`, which the tests verify by brute
     force over all pairs.
   - Raise on single-class input (AUC is undefined, not 0 or 1).

2. **PR curve + AP (`precision_recall_curve`, `average_precision`)**
   - Sort by score descending; cumulative TP/FP give precision/recall at
     every depth.
   - `AP = Σ (Rᵢ − Rᵢ₋₁) · Pᵢ` — the step integral, not trapezoidal.

3. **ECE (`expected_calibration_error`)**
   - Bin by confidence, weighted mean of |bin accuracy − bin confidence|.

---

## Behavior Notes / Gotchas

- **Ties are the AUC landmine.** Real scores collide constantly (quantized
  models, coarse features). Counting ties as wins or losses biases AUC;
  average ranks give each tied pair exactly ½. The test uses coarsened
  scores to force collisions and checks against the O(n²) pairwise
  definition to 1e-12.
- **AUC is a pure ranking metric** — invariant to any strictly monotone
  transform of the scores (tested with a sigmoid and a cube). Corollary:
  AUC says *nothing* about calibration.
- **AUC vs AP under imbalance.** Diluting the same ranking with 10× easy
  negatives leaves AUC ~unchanged (it credits true negatives via FPR) while
  AP drops sharply (it never looks at true negatives). For rare-positive
  problems — fraud, safety filters, retrieval — report AP. Both behaviors
  are tested on the same data.
- **Good AUC + bad calibration is common** — e.g. scores squashed into
  [0.45, 0.55] rank perfectly but are wildly underconfident. Post-hoc fixes
  (temperature scaling, isotonic) change ECE without moving AUC at all.
- **ECE testing is statistical:** a perfectly calibrated synthetic model
  (labels drawn with probability = prediction) must score ≈ 0; a
  0.95-confidence/60%-accuracy model must score ≈ 0.35.

---

## Running the Smoke Test

```bash
pip install numpy pytest
python -m pytest test_metrics.py -v
```

| Test | Validates |
|------|-----------|
| `test_auc_perfect_reversed_random` | 1.0 / 0.0 / ≈0.5 anchors |
| `test_auc_equals_pairwise_probability` | Rank formula ≡ pairwise definition (with ties) |
| `test_auc_tie_handling_hand_case` | All-ties ⇒ exactly 0.5 |
| `test_auc_invariant_to_monotone_transform` | Ranking-only property |
| `test_auc_requires_both_classes` | Undefined-input handling |
| `test_pr_curve_hand_case` | Hand-computed precision/recall/AP |
| `test_ap_perfect_ranking_is_one` | AP anchor |
| `test_ap_reflects_imbalance_but_auc_does_not` | The metric-choice lesson |
| `test_ece_perfectly_calibrated_is_small` | ECE ≈ 0 for calibrated model |
| `test_ece_overconfident_model_is_flagged` | ECE ≈ gap for overconfident model |
| `test_good_auc_bad_calibration` | AUC and ECE measure different things |

---

## Discussion Questions (interview follow-ups)

- **Derive** AUC = P(s₊ > s₋) from the ROC-curve-area definition.
- **Confidence intervals** — how do you bootstrap an AUC CI, and why is the
  naive per-sample bootstrap wrong for grouped data (per-user examples)?
- **Threshold selection** — given asymmetric costs, pick the operating point
  from the PR curve; why is maximizing F1 usually the wrong default?
- **Calibration under shift** — a model calibrated on the training
  distribution meets covariate shift. What happens to ECE vs AUC?
- **LLM evals** — how do these ideas map to LLM-judge scores (ties
  everywhere), pass@k (AP-like), and verbalized-confidence calibration?
