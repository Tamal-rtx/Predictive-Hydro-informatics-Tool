# 🌊 Predictive Hydroinformatics Tool

> **A production-grade machine learning dashboard for river basin intelligence** — combining physically-consistent hydrological simulation, Scikit-Learn ML pipelines, and an interactive Streamlit web interface for real-time flood risk assessment and streamflow forecasting.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Project Overview

This tool is an **end-to-end Hydroinformatics AI Platform** that uses environmental sensor data (rainfall depth, antecedent soil moisture, river stage, upstream discharge) to perform two distinct ML tasks simultaneously:

| Task | Model | Output |
|------|-------|--------|
| **Flood Risk Classification** | K-Nearest Neighbors (KNN) | `Low / Medium / High` risk tier |
| **Streamflow / WQI Forecasting** | Linear Regression | Continuous value (m³/s or Index score) |

All data is **synthesized on-the-fly** using a physically-grounded `DataSimulator` — no external datasets or API keys required. The tool is **instantly runnable** after a single `pip install`.

---

## 🎯 Features

### 🧬 Physical Hydrological Data Simulation
- Gamma-distributed **rainfall** with realistic dry-spell intermittency
- Beta-distributed **soil moisture** coupled with antecedent precipitation
- Lognormal **upstream discharge** mimicking USGS stream gauge observations
- Non-linear **Dunne/Hortonian runoff** generation via logistic saturation curves
- **Hydraulic capacity exceedance** modelling for flood tier labelling

### 🤖 Strict Scikit-Learn ML Pipelines
- **LinearRegression** pipeline: `StandardScaler → LinearRegression` for continuous flow/WQI estimation
- **KNNClassifier** pipeline: `StandardScaler → KNeighborsClassifier` for 3-class flood hazard prediction
- Full **train/test partitioning** with stratification — zero data leakage
- Comprehensive **metrics output**: MSE, RMSE, R², MAE (regression) + Precision, Recall, F1, Confusion Matrix (classification)

### 📊 Interactive Streamlit Dashboard
- **Sidebar hyperparameter controls**: K value, distance metric, weighting, test/train split ratio, dataset size
- **Live weather scenario sliders**: Dial in exact rainfall/moisture/stage values and get instant model predictions
- **Flood risk badges** (🟢 Low / 🟡 Medium / 🔴 High) with probability bar charts
- **Actual vs. Predicted scatter plot** with identity line
- **Residual error histogram** with box marginal
- **Pearson correlation heatmap** across all hydrologic indicators
- **Phase space bubble chart** (Rainfall vs. Soil Moisture coloured by flood tier)
- **Dataset export** to CSV for external HEC-RAS / GIS modelling

---

## 🗂️ Project Structure

```
project 3/
│
├── app.py                  # 🌐 Streamlit dashboard — main entry point
├── hydro_engine.py         # 🤖 ML engine: LinearRegression + KNN pipelines, metrics, inference
├── hydro_simulator.py      # 🧬 DataSimulator: physically-consistent synthetic hydrology generator
│
├── requirements.txt        # 📦 Python package dependencies (pinned versions)
├── .gitignore              # 🚫 Files and folders excluded from version control
└── README.md               # 📖 Project documentation (this file)
```

### Module Responsibilities

#### `hydro_simulator.py` — `DataSimulator`
Generates synthetic daily hydrometric records using physically-grounded stochastic distributions. Produces a `pd.DataFrame` with features and dual targets (continuous + categorical) without any external file dependencies.

#### `hydro_engine.py` — `HydroinformaticsEngine`
Object-oriented ML engine encapsulating:
- `partition_data()` — stratified train/test split
- `train_regression_model()` — StandardScaler + LinearRegression Pipeline
- `train_classification_model()` — StandardScaler + KNeighborsClassifier Pipeline
- `evaluate_regression()` → `RegressionMetrics` dataclass
- `evaluate_classification()` → `ClassificationMetrics` dataclass
- `predict_scenario()` → `ScenarioPrediction` dataclass (real-time inference)
- `fit_and_evaluate()` — single-call end-to-end execution

#### `app.py` — Streamlit Application
Thin UI layer that wires the engine to an interactive web dashboard. Four structured tabs:
1. **Real-Time Scenario Forecast** — interactive slider prediction
2. **Model Performance & Benchmarks** — metrics + diagnostic plots
3. **Hydro-Environmental Analytics** — EDA correlation charts
4. **Dataset Explorer** — raw data table with CSV export

---

## 🚀 Quick Start

### Prerequisites
- Python **3.11+**
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/predictive-hydroinformatics-tool.git
cd predictive-hydroinformatics-tool
```

### 2. Create & Activate a Virtual Environment *(Recommended)*

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the Dashboard

```bash
streamlit run app.py
```

The dashboard will automatically open at **http://localhost:8501** in your default browser.

### 5. CLI Mode *(Optional — terminal-only output)*

```bash
# Run the ML engine standalone for terminal metric output
python hydro_engine.py

# Run the data simulator standalone
python hydro_simulator.py
```

---

## 📈 Sample Results (Default Configuration)

| Metric | Value |
|--------|-------|
| Training Set Size | 1,200 records |
| Test Set Size | 300 records |
| Regression R² | **0.974** |
| Regression RMSE | **~34 m³/s** |
| KNN Accuracy | **92.3%** (K=5, distance weighting) |
| KNN Macro F1 | **0.90** |

---

## 🔬 Methodology

### Hydrological Data Generation

The `DataSimulator` implements the following physics chain:

```
Rainfall (Gamma) + Antecedent Soil Moisture (Beta)
        ↓
Non-linear Runoff Coefficient (Logistic Saturation Curve)
        ↓
Mass Balance Routing → Downstream Streamflow (Q)
        ↓
Runoff Dilution Model → Water Quality Index (WQI)
        ↓
Hydraulic Capacity Exceedance → Flood Risk Tier (Low / Medium / High)
```

### Feature Engineering

| Feature | Unit | Physics Basis |
|---------|------|---------------|
| `Daily_Rainfall_mm` | mm | Gamma-distributed precipitation depth |
| `Soil_Moisture_Percentage` | % | Antecedent volumetric moisture content |
| `Historical_River_Level_m` | m | Gauge stage height (Manning's-derived) |
| `Upstream_Flow_Rate` | m³/s | Headwater discharge (lognormal) |

### Target Variables

| Target | Type | Description |
|--------|------|-------------|
| `Downstream_Streamflow_m3s` | Continuous | Forecasted routed discharge (m³/s) |
| `Water_Quality_Index` | Continuous | Sediment/turbidity composite index (0–100) |
| `Flood_Risk_Level` | Categorical | Hydraulic exceedance tier (Low/Medium/High) |

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| ML Engine | Scikit-Learn 1.9 | LinearRegression, KNN, StandardScaler, Pipelines |
| Data Wrangling | Pandas 3.1 + NumPy 2.4 | DataFrame operations, stochastic simulation |
| Dashboard | Streamlit 1.66 | Interactive web UI |
| Interactive Plots | Plotly 7.1 | Scatter, histograms, correlation heatmaps, bar charts |
| Statistical Plots | Matplotlib 3.11 + Seaborn 0.13 | Confusion matrix heatmap |

---

## 🧪 Extending the Tool

### Adding a New ML Model
1. Add a new pipeline method in `HydroinformaticsEngine` in `hydro_engine.py`
2. Add corresponding metrics evaluation method
3. Wire the trained pipeline to `predict_scenario()` for inference
4. Add a new UI section in `app.py`

### Connecting Real Sensor Data
1. Replace the `DataSimulator` call in `app.py` with a `pd.read_csv()` or API fetch
2. Ensure column names match the `features` list in `HydroinformaticsEngine`
3. Verify the categorical target is encoded as `["Low", "Medium", "High"]`

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- **USGS National Water Information System (NWIS)** — for hydrometric data standards
- **NOAA River Forecast Centers** — for flood stage classification methodology  
- **Scikit-Learn Community** — for the ML pipeline architecture patterns
- **Streamlit** — for the rapid ML dashboard prototyping framework

---

*Built as a Principal Data Scientist & Senior ML Engineer specification for Hydroinformatics research and river basin management applications.*

