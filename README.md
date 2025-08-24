#  Grade Predictor

## ✨ Features

* Trains a machine‑learning model on `student_performance.csv`.
* Saves ready‑to‑use artifacts: `grade_model.pkl`, `gender_encoder.pkl`, `support_encoder.pkl`.
* Simple app (`app.py`) to load the model and serve predictions (lightweight Flask setup).
* Clean preprocessing: categorical features are encoded and reused at inference.
* Re‑train anytime with your own CSV (same columns) to update the model.
* Uses familiar, lightweight libraries: `pandas`, `numpy`, `scikit‑learn`, `joblib`, `flask`.

---

## 🛠️ Installation (Step‑by‑Step)

1. **Install Python**

   * Use Python **3.9+**.

2. **Clone the repository**

```bash
git clone https://github.com/rahuldadige/grade-predictor.git
cd grade-predictor
```

3. **Create & activate a virtual environment (recommended)**

```bash
# create
python -m venv .venv

# activate
# Windows
. .venv/Scripts/activate
# macOS/Linux
source .venv/bin/activate
```

4. **Install dependencies**

```bash
pip install -U pandas numpy scikit-learn joblib flask
```

5. **(Optional) Retrain the model**

```bash
python model.py
```

This reads `student_performance.csv` and writes fresh `.pkl` artifacts.

6. **Run the app**

```bash
python app.py
```

The app will start locally (default Flask port `http://127.0.0.1:5000`).

> That’s it—installed and running. If you switch to a new dataset, keep the same column names to reuse the pipeline.
