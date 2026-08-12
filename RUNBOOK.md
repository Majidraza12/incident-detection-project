# Runbook — reproduce the results and fill in the README

Step-by-step for **Windows PowerShell**. Run everything from the project root:
`C:\College\IIT 2026 Spring\CS-495\project\incident-detection-project`.

Every command below uses the project venv Python directly
(`.venv\Scripts\python.exe`), so you do **not** need to activate anything. If you
prefer to activate the venv once instead, run `.\.venv\Scripts\Activate.ps1` and
then you can type `python` in place of `.\.venv\Scripts\python.exe`.

> **Time budget (this machine, CPU-only): ~0.8 s per clip/step.**
> Step 2 ≈ 2 hours, Step 4 ≈ 2 hours (both scripts run 5 epochs × 3 seeds).
> Note this venv has a **CPU-only torch build** (`2.11.0+cpu`), so it will not
> use a GPU even on a machine that has one — that would need a CUDA torch
> install first.

---

## Step 0 — Sanity check (30 seconds)

Confirm the environment is wired up before committing hours.

```powershell
.\.venv\Scripts\python.exe -c "import torch; print('torch', torch.__version__, '| GPU:', torch.cuda.is_available())"
```

- **Expect:** a torch version prints. `GPU: True` means the long steps will be
  fast; `GPU: False` means budget the hours above.

---

## Step 1 — Confirm the splits exist (already done — skip unless missing)

The `data/splits/` folders already exist. Only run this if they're gone:

```powershell
.\.venv\Scripts\python.exe src\make_splits.py
```

- **Verify:** these should print non-zero counts.

```powershell
Get-ChildItem -Recurse data\splits -Filter *.mp4 | Group-Object { $_.Directory.Parent.Name + '/' + $_.Directory.Name } | Select-Object Count, Name
```

- **Expect:** train/violent ≈ 161, train/non-violent ≈ 84, val ≈ 34/18, test ≈ 35/18.

---

## Step 2 — Run the temporal ablation (~2 hours on CPU)

Trains 2 models × 3 seeds × 5 epochs (CNN-LSTM vs frame-only mean-pool), each
saving its best-by-validation-loss checkpoint.

```powershell
.\.venv\Scripts\python.exe src\train_ablation.py
```

- **What you'll see:** per-epoch lines like
  `[cnn_lstm seed0] Epoch 3/10 | Train Loss ... Val Loss ...  <- best (saved)`.
  The `<- best (saved)` marker shows which epoch's weights were kept.
- **Tip:** to run unattended and keep a log:
  ```powershell
  .\.venv\Scripts\python.exe src\train_ablation.py *>&1 | Tee-Object outputs\ablation_run.log
  ```
- **Verify when done:**
  ```powershell
  Get-ChildItem outputs\checkpoints\ablation
  Get-Content outputs\metrics\ablation\runs_summary.csv
  ```
  **Expect:** 6 `.pth` files (`cnn_lstm_seed0/1/2`, `frame_mean_seed0/1/2`) and a
  `runs_summary.csv` with 6 rows.

---

## Step 3 — Build the ablation table (~2 minutes)

Evaluates all 6 checkpoints on the 53-clip test set and aggregates mean ± std.

```powershell
.\.venv\Scripts\python.exe src\make_ablation_table.py
```

- **Verify:**
  ```powershell
  Get-Content outputs\metrics\ablation\ablation_table.md
  Get-Content outputs\metrics\ablation\ablation.csv
  ```
- **Then paste** the contents of `ablation_table.md` into `README.md`, replacing
  the placeholder table under **"Temporal ablation — does the LSTM help?"** (the
  block with `_run to fill_`).

---

## Step 4 — Overfitting diagnostic (~2 hours on CPU) — optional but recommended

Compares the train–val accuracy gap of the status-quo model vs an augmented +
frozen-backbone variant.

```powershell
.\.venv\Scripts\python.exe src\train_overfit_probe.py
```

- **Verify:**
  ```powershell
  Get-Content outputs\metrics\overfit_probe\overfit_probe.csv
  ```
  **Expect:** the `aug_frozen` rows to show a **smaller** `train_val_gap` than
  `original` — that's the evidence the overfitting was addressed. Mention the two
  gap numbers in the README's "Overfitting diagnostic" section.

---

## Step 5 — Single-model report + confusion matrix (~3 minutes)

Regenerates `classification_report.txt` and `confusion_matrix.png` (already in the
repo; rerun only if you want them fresh).

```powershell
.\.venv\Scripts\python.exe src\evaluate.py
```

- **Verify:** `outputs\plots\confusion_matrix.png` updated; report prints to console.

---

## Step 6 — Demo (any time)

```powershell
# Web app: opens in your browser, upload a clip
.\.venv\Scripts\python.exe -m streamlit run src\app.py

# CLI: one video
.\.venv\Scripts\python.exe src\predict.py data\splits\test\violent\<some>.mp4
```

---

## If you want real numbers faster (trade rigor for speed)

Edit the top of `src\train_ablation.py`:

- **Fewer epochs — already applied.** `EPOCHS` is now `5` (was 10). Justified by
  the prior runs: best validation loss landed at **epoch 1** in both, so later
  epochs never produced a kept checkpoint.
- **Fewer seeds:** change `SEEDS = [0, 1, 2]` to `SEEDS = [0]` → ~1/3 the time, but
  you lose the mean ± std (report a single number and say so). Prefer cutting
  epochs over seeds — with one seed you cannot tell a real LSTM-vs-mean-pool
  difference from seed noise on a 53-clip test set.

Both keep the comparison fair as long as you change them for **both** arms (they're
global, so you do). Re-run Steps 2–3 after editing.

---

## Reproducibility check (optional, proves the seeding works)

Run the ablation twice and confirm identical per-epoch metrics for a given seed —
compare two `history_cnn_lstm_seed0.csv` files from separate runs; they should
match line for line.
