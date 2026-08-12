# Project Overview — Video-Based Incident (Violence) Detection

One-pass read: what this project is, what was built, what the numbers are, and
what is honestly still pending. Written for interview prep and resume framing.

**Repo:** CS-495 final project, IIT, Spring 2026 · Python / PyTorch / Streamlit

---

## 1. What it does

Give it a short surveillance-style video clip; it returns **violent** or
**non-violent** plus a confidence score.

Pipeline per clip:

1. Sample **16 evenly-spaced frames** across the clip.
2. Push each frame through a **pretrained ResNet-18** → 512-dim appearance feature.
3. Feed the 16-frame sequence into an **LSTM** → models motion over time.
4. Classify from the final LSTM timestep → label + softmax confidence.

Two interfaces: a **Streamlit web app** (upload → predict) and a **CLI**
(`python src/predict.py clip.mp4`).

---

## 2. Headline numbers

Evaluated on a held-out **53-clip test set** (35 violent / 18 non-violent):

| | precision | recall | f1-score | support |
|---|---:|---:|---:|---:|
| non-violent | 0.88 | 0.78 | 0.82 | 18 |
| violent | 0.89 | 0.94 | 0.92 | 35 |
| **accuracy** | | | **0.89** | 53 |
| macro avg | 0.88 | 0.86 | 0.87 | 53 |

**→ 89% accuracy, 0.92 F1 on the violent class.**

Data: ~350 clips total, split 245 train / 52 val / 53 test, with a ~2:1 class
imbalance toward violent.

---

## 3. What changed in this round of work

The model was already working. This round was about **experimental rigor** —
turning "a model that scores 89%" into "a project that can defend that 89%."

### Built and working

| Change | File | Why it matters |
|---|---|---|
| **Reproducibility harness** | `train_utils.py` | `set_seed` covers Python/NumPy/torch RNGs + cuDNN determinism, and a seeded DataLoader generator makes shuffling repeatable. Same seed → identical per-epoch metrics. |
| **Best-by-validation-loss checkpointing** | `train_utils.py:train_run` | Saves the checkpoint from the epoch with the lowest validation loss, not the last epoch. On this dataset that is a large difference (see below). |
| **Frame-only baseline model** | `model.py:CNNMeanPool` | Same ResNet-18 backbone, but mean-pools the 16 frame features instead of using an LSTM. Isolates whether temporal modeling actually earns its keep. |
| **Data augmentation + frozen-backbone option** | `dataset.py` | Train-only augmentation and ImageNet normalization; backbone can be frozen so only the LSTM and head train. |
| **Cached model loading** | `predict.py` | Was reloading ~48 MB of weights on every single call. |
| **Refactored evaluation** | `evaluate.py` | Extracted a reusable `evaluate_checkpoint(...)` so any checkpoint can be scored on the test set, not just the one hardcoded path. |
| **Honest README** | `README.md` | Results, method, reproduction steps, and a limitations section that names the problems rather than hiding them. |

### The key empirical finding

From the logged training runs (`outputs/metrics/training_metrics*.csv`), across
**two independent runs**:

- **Validation loss is lowest at epoch 1** — 0.449 and 0.429 — and never improves
  again through epoch 10.
- Meanwhile training loss collapses to **0.003** and training accuracy hits **1.0**.

The model memorizes the 245 training clips in essentially one pass. Everything
after epoch 1 is overfitting. This is a real, measured result, and it drove a
concrete engineering decision: training was cut from 10 epochs to 5, since
epochs 2–10 never produced a checkpoint that was kept.

**This is the single best thing to talk about in an interview.** It shows you
read your own telemetry and changed the design because of it.

### Built but not yet run

Two experiments are fully implemented and waiting on ~4 hours of CPU time:

1. **Temporal ablation** (`train_ablation.py` → `make_ablation_table.py`) —
   CNN-LSTM vs. frame-only mean-pool, 2 models × 3 seeds, identical splits,
   seeds, and hyperparameters. Answers: *does the LSTM actually beat frame
   averaging?*
2. **Overfitting probe** (`train_overfit_probe.py`) — measures the train–val
   accuracy gap of the current setup against an augmented + frozen-backbone
   variant. Answers: *did the fix actually shrink the gap?*

The README marks both as pending rather than showing empty cells.

---

## 4. Honest limitations (be ready for these)

- **Small test set (53 clips).** One clip = 1.9% accuracy. 89% has a wide
  confidence interval — this is a proof of concept, not a validated production
  metric. Say so before an interviewer does.
- **Small dataset (~350 clips) vs. a fine-tuned 12M-parameter model.** Hence the
  memorization described above.
- **Possible split leakage.** `make_splits.py` splits at the *clip* level. Clips
  from the same scene with sequential filenames could land in different splits,
  which would inflate test scores. Known caveat, not solved.
- **Binary and short-clip only.** No temporal localization (*when* in a long
  video), no multi-class incident types, no real-time streaming.
- **Only the final LSTM timestep is used.** Attention or mean-pooling over all
  timesteps is the natural next improvement.

---

## 5. Good "what would you do next?" answers

Attention over all timesteps instead of just the last; a scene-disjoint
(leakage-free) split; more data before more model capacity; temporal
localization; multi-class incident types.

---

## 6. Repo map

```
src/
  model.py               # CNNLSTM + CNNMeanPool (ablation baseline)
  dataset.py             # 16-frame sampling, optional augmentation + ImageNet norm
  train_utils.py         # set_seed, seeded generator, train_run (best-by-val)
  train.py               # original single-run trainer
  train_ablation.py      # 2 models x 3 seeds -> best-by-val checkpoints
  make_ablation_table.py # evaluate checkpoints -> ablation.csv + .md
  train_overfit_probe.py # augmentation + frozen-backbone diagnostic
  evaluate.py            # evaluate_checkpoint(...) + classification report
  predict.py / app.py    # CLI + Streamlit inference
outputs/                 # checkpoints, metrics, plots
data/splits/             # train / val / test
```
