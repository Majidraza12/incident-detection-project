# Project Context — Video-Based Incident (Violence) Detection

> A reference/context document capturing what this project is, what was actually
> built, and its honest strengths and weaknesses. Written to help decide how (and
> whether) to present it on a resume, and to feed context to future conversations.

---

## 1. One-line summary

A deep-learning system that classifies short surveillance-style video clips as
**violent** or **non-violent** using a **CNN-LSTM** architecture, delivered
end-to-end from raw data through a live web demo.

- **Course:** CS-495 (final semester project), IIT, Spring 2026
- **Language / stack:** Python, PyTorch, torchvision, OpenCV, scikit-learn, Streamlit
- **Status:** Complete. Demo delivered successfully.

---

## 2. What the project actually does

Given a video clip, the system:
1. Samples **16 evenly-spaced frames** across the clip's duration.
2. Runs each frame through a **pretrained ResNet-18** to extract a 512-dim
   appearance feature.
3. Feeds the 16-frame feature sequence into an **LSTM** to model motion/temporal
   dynamics.
4. Classifies the sequence (from the final LSTM timestep) as **violent** or
   **non-violent**, returning a label and a softmax confidence score.

It is usable two ways:
- **CLI:** `python src/predict.py path/to/video.mp4`
- **Web app:** `streamlit run src/app.py` — upload a clip, get a prediction with confidence.

---

## 3. Architecture (the model)

**CNN-LSTM** (`src/model.py`):
- **Backbone:** `resnet18(weights=ResNet18_Weights.DEFAULT)`, final FC replaced with
  `Identity()` → 512-dim per-frame feature.
- **Temporal model:** single-layer `nn.LSTM`, input 512, hidden 256, `batch_first=True`.
- **Head:** `Dropout(0.3)` → `Linear(256 → 2)`.
- **Forward pass:** input `[B, T=16, C, H, W]` is flattened to `[B*T, C, H, W]` for
  the CNN, reshaped back to `[B, T, 512]` for the LSTM, and the **last timestep**
  feeds the classifier.

Why this is a reasonable design: violence is defined by both *appearance* (what's
in the frame) and *motion over time* (how it changes). CNN-per-frame + LSTM-over-time
captures both — a textbook-appropriate choice for video classification.

---

## 4. Data

- **Dataset:** a violence-detection video dataset (`data/processed/violence-detection-dataset/`),
  organized into `violent/` and `non-violent/` folders of `.mp4` clips.
- **Raw processed totals:** ~**230 violent**, ~**120 non-violent** (≈350 clips).
- **Train / Val / Test splits** (`data/splits/`):

  | Split | Violent | Non-violent | Total |
  |-------|--------:|------------:|------:|
  | Train | 161     | 84          | 245   |
  | Val   | 34      | 18          | 52    |
  | Test  | 35      | 18          | 53    |

- **Class imbalance** (~2:1 toward violent) is handled explicitly via
  **inverse-frequency class weighting** in the loss function.

---

## 5. Training setup (`src/train.py`)

- **Loss:** `CrossEntropyLoss` with computed class weights.
- **Optimizer:** Adam, LR = 1e-4.
- **Epochs:** 10, **batch size:** 2.
- **Device:** CUDA if available, else CPU.
- **Backbone is fine-tuned end-to-end** (not frozen).
- Saves checkpoint (`cnn_lstm_weighted.pth`), a metrics CSV, and loss/accuracy plots.

---

## 6. Results (`outputs/metrics/classification_report.txt`)

Evaluated on the **53-clip test set** (`src/evaluate.py`, also produces a confusion matrix):

|              | precision | recall | f1-score | support |
|--------------|----------:|-------:|---------:|--------:|
| non-violent  | 0.88      | 0.78   | 0.82     | 18      |
| violent      | 0.89      | 0.94   | 0.92     | 35      |
| **accuracy** |           |        | **0.89** | 53      |
| macro avg    | 0.88      | 0.86   | 0.87     | 53      |
| weighted avg | 0.89      | 0.89   | 0.89     | 53      |

**Headline number: ~89% accuracy, 0.92 F1 on the violent class.**

---

## 7. Repository structure (what was built)

```
src/
  model.py             # CNN-LSTM architecture
  dataset.py           # PyTorch Dataset: frame sampling + preprocessing
  train.py             # Training loop, class weighting, metrics + plots
  evaluate.py          # Test-set evaluation: classification report + confusion matrix
  predict.py           # CLI single-video inference
  batch_predict.py     # Batch inference
  app.py               # Streamlit web demo (upload → predict)
  make_splits.py       # Train/val/test split creation
  preprocess_videos.py # Video preprocessing
  dataset_stats.py     # Dataset statistics
  inspect_dataset.py   # Dataset inspection
  test_dataset.py      # Dataset sanity checks
  test_model.py        # Model sanity checks
  plots/               # Standalone plotting scripts (loss, accuracy, test)
outputs/
  checkpoints/         # Saved model weights
  metrics/             # classification_report.txt, training CSVs
  plots/               # Loss/accuracy curves, confusion matrix
data/
  processed/           # Raw clips by class
  splits/              # train / val / test
requirements.txt
```

This is a **complete ML pipeline**, not just a notebook: data prep → splits →
training → evaluation → inference → deployable demo.

---

## 8. Strengths (resume-relevant)

- **Correct, well-motivated architecture.** CNN-LSTM is the right tool; shows
  understanding of both spatial and temporal modeling, not just plugging in a classifier.
- **Real, respectable result.** ~89% accuracy / 0.92 violent-class F1.
- **End-to-end and demoable.** Streamlit app means it can be shown live — strong for
  interviews. Most student projects stop at a notebook; this ships a UI.
- **Engineering maturity signals:** proper train/val/test discipline, class-imbalance
  handling via weighted loss, separate evaluation with confusion matrix, transfer
  learning from a pretrained backbone, saved checkpoints and reproducible metrics.
- **Clean, modular code** — each concern in its own file.
- **Socially meaningful application** (public-safety / surveillance), which makes for
  a good talking point.

---

## 9. Weaknesses / honest caveats (be ready to address these)

- **Small test set (53 clips).** 89% is encouraging but has a wide confidence
  interval; it's a proof of concept, not a validated production metric. **Frame it that way.**
- **Small dataset overall (~350 clips)** while fine-tuning a full ResNet-18 end-to-end
  → meaningful **overfitting risk**. Worth checking the train-vs-val loss curves and
  being honest about it. (Freezing the backbone or heavier augmentation would be the
  natural next step.)
- **No data augmentation** in the dataset pipeline (only resize + tensor).
- **`predict.py` reloads the model on every call** — fine for a demo, inefficient for
  real/batch use.
- **Empty README.** The single biggest, cheapest fix — a finished-looking repo needs one.
- **Binary + short clips only.** No temporal localization ("when" in a long video),
  no multi-class incident types, no real-time streaming.
- **Only the final LSTM timestep is used** — attention or mean-pooling over timesteps
  could help, and is a good "what I'd do next" answer.

---

## 10. Resume verdict & framing

**Yes — this is resume-worthy**, provided it's framed as an applied deep-learning /
computer-vision project rather than novel research. It demonstrates the full ML
lifecycle and produces a working, demoable artifact, which is exactly what employers
look for in a student CV.

**Suggested resume bullet (draft):**
> Built an end-to-end video violence-detection system (CNN-LSTM: ResNet-18 + LSTM in
> PyTorch) achieving 89% accuracy / 0.92 F1 on held-out clips; handled class imbalance
> with weighted loss and shipped a live Streamlit demo for upload-and-classify inference.

**Open question for the resume decision:** how this ranks against the other projects
already on / planned for the resume. (Assistant does **not** currently have context on
those other projects — user to provide them for a comparison.)

---

## 11. Suggested next steps (quick wins, if desired)

1. Write a real **README** (what it does, the 89% result, how to run the app).
2. Add basic **data augmentation** + try **freezing the ResNet backbone** to test
   overfitting sensitivity.
3. Cache the model in `predict.py` (load once).
4. Add the **confusion matrix** and loss/accuracy curves to the README as visuals.
