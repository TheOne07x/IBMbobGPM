# 🛰️ Global Precipitation Measurement (GPM) Rainfall Engine

> **AI-Powered Planetary Hydro-Climatic Intelligence & Risk Simulation Dashboard**

---

**Author:** Almas Qureshi  
**Programme:** IBM SkillsBuild Data Analytics with AI — Academic Virtual Internship  
**In association with:** CSRBOX / BharatCares & AICTE  

---

## 📋 Project Overview

The **GPM Rainfall Engine** is a production-grade, end-to-end machine learning and interactive telemetry dashboard that forecasts national rainfall, classifies water availability risk, and simulates future climate scenarios in real time.

Built on a **Gradient Boosting Regressor** trained on the Kaggle Climate Change Indicators dataset, the system:

- Ingests and engineers multi-year sovereign climate records (2000–2023)
- Derives per-country hydrological risk classifications — **Low (Water Difficulty)**, **Decent (Normal)**, and **Heavy (Surplus)** — using Z-score normalisation
- Projects rainfall for any year up to **2030** via extrapolated climate drivers
- Delivers an interactive **Streamlit command-centre dashboard** with a global choropleth map, sovereign deep-dive dossiers, dual-axis trend overlays, and a live **What-If Climate Shock Simulator** with dynamic impact briefings

---

## 📂 Dataset Reference

| Field       | Detail |
|-------------|--------|
| **Source**  | Kaggle — Climate Change Indicators Dataset |
| **URL**     | https://www.kaggle.com/datasets/bhadramohit/climate-change-dataset |
| **Records** | ~500+ rows across 15 countries, 2000–2023 |
| **Key Columns** | Year, Country, Avg Temperature (°C), CO2 Emissions (Tons/Capita), Sea Level Rise (mm), Rainfall (mm), Population, Renewable Energy (%), Extreme Weather Events, Forest Area (%) |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| Dashboard | Streamlit ≥ 1.35 |
| Visualisation | Plotly Express & Graph Objects ≥ 5.22 |
| ML Framework | Scikit-Learn ≥ 1.4 |
| Data Processing | Pandas ≥ 2.2, NumPy ≥ 1.26 |
| Model Persistence | Joblib ≥ 1.4 |
| Report Generation | python-docx ≥ 1.1 |

---

## 🗂️ Project Architecture

```
IBM Project GPM/
│
├── climate_change_dataset.csv       # Source dataset (Kaggle)
│
├── train_model.py                   # ML pipeline: clean → engineer → train → save
├── app.py                           # Streamlit interactive dashboard
├── generate_report.py               # python-docx report generator
│
├── requirements.txt                 # Pinned Python dependencies
├── README.md                        # This file
│
├── AlmasQureshi_GlobalRainfallEngine.ipynb   # Jupyter analysis notebook
├── AlmasQureshi_ProjectReport.docx           # Auto-generated project report
│
├── model_artifacts/                 # Generated after running train_model.py
│   ├── rainfall_model.pkl           # Trained Gradient Boosting pipeline
│   └── feature_meta.pkl             # Country stats, feature list, processed df
│
└── screenshots/                     # Dashboard screenshots (optional)
```

---

## ⚙️ Setup & Execution Guide

### 1 — Clone / open the project folder

```bash
cd "IBM Project GPM"
```

### 2 — Create a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\activate             # Windows
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### 4 — Train the ML model

This step loads the dataset, engineers features, trains the Gradient Boosting model, prints evaluation metrics, and saves artefacts to `model_artifacts/`.

```bash
python train_model.py
```

Expected output:
```
[load]  500+ valid rows | 15 countries | Years 2000–2023
[label] Water status distribution: {...}
[feats] Feature columns: [...]
[train] Training on ~400 rows …
[train] Training complete.
═══════════════════════════════════════════════
  MODEL EVALUATION (held-out 20%)
═══════════════════════════════════════════════
  RMSE :     xxx.xx mm
  MAE  :     xxx.xx mm
  R²   :     0.xxxx
═══════════════════════════════════════════════
[save]  Model  → model_artifacts/rainfall_model.pkl
[save]  Meta   → model_artifacts/feature_meta.pkl
[done]  All artifacts saved. Run `streamlit run app.py` to launch.
```

### 5 — Launch the dashboard

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

### 6 — Generate the project report (optional)

```bash
python generate_report.py
```

Produces `AlmasQureshi_ProjectReport.docx` in the project root.

---

## 🌟 Core Feature Breakdown

### 🗺️ Global Choropleth Water Anomaly Matrix (2000–2030)
A full-bleed, dark-mode Plotly choropleth map colour-codes every tracked nation by its water availability status for any selected year. Hovering reveals rainfall (mm), temperature, and CO₂ data. For years beyond the dataset (2025–2030), the system generates AI-projected values using the trained model and extrapolated climate drivers.

### 🏛️ Sovereign Climate Intelligence & Risk Dossier
Selecting any nation reveals a four-card telemetry panel showing:
- Historical baseline rainfall (all-time country mean)
- Mean temperature and 20-year warming trend (°C/decade via OLS regression)
- Average annual extreme weather frequency
- Water Vulnerability Index (Low / Moderate / High Risk, derived from historical deficit frequency)

### 📈 Dual-Axis Trends & 3-Year Rolling Precipitation Analysis
A precipitation trajectory chart overlays:
- Recorded annual rainfall (area fill)
- 3-year rolling moving average (amber dotted line)
- Historical baseline reference (dashed marker)
- ◆ AI forecast diamond for the selected target year

A secondary dual-axis chart overlays extreme weather event frequency (bar) against average temperature trend (line) per year.

### 🧪 Interactive Climate Shock Simulator (What-If Engine)
Two interactive sliders — **Simulated Temp Anomaly (+°C)** and **Simulated Extreme Weather Events / Year** — feed live inputs to the trained ML model. Results are displayed as:
- A large rainfall forecast figure with percentage delta from the historical baseline
- A coloured water status badge (Low / Decent / Heavy)
- A **dynamic impact briefing** covering agricultural drought risk, monsoon variability, flood defence readiness, and groundwater outlook

---

## 📊 Model Details

| Parameter | Value |
|-----------|-------|
| Algorithm | Gradient Boosting Regressor (sklearn) |
| Trees | 300 |
| Learning Rate | 0.05 |
| Max Depth | 5 |
| Subsample | 0.85 |
| Scaler | StandardScaler |
| Train/Test Split | 80% / 20% |
| Features | Avg Temp, CO₂ Emissions, Extreme Weather Events, Rainfall Lag-1, Rainfall Lag-2, Rainfall 3-yr Rolling Mean |
| Target | Rainfall (mm) |

---

## 📄 License

This project was created for academic and internship purposes under the IBM SkillsBuild programme.  
Dataset sourced from Kaggle under the applicable open data licence.
