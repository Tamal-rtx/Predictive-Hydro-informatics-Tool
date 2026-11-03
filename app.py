"""Predictive Hydroinformatics Tool - Streamlit Application.

An interactive dashboard for real-time flood risk classification (KNN)
and streamflow / water quality regression (Linear Regression)
using physically-consistent environmental and hydrological data.
"""

from typing import Dict, Any, List, Tuple
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from hydro_simulator import DataSimulator
from hydro_engine import HydroinformaticsEngine, RegressionMetrics, ClassificationMetrics


# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Hydroinformatics AI Platform",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Hydro-Themed UI Cards and Badges
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0b4f6c;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f7fafc;
        border-left: 5px solid #0288d1;
        border-radius: 8px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        margin-bottom: 15px;
    }
    .risk-badge-low {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #a5d6a7;
    }
    .risk-badge-medium {
        background-color: #fff8e1;
        color: #f57f17;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #ffe082;
    }
    .risk-badge-high {
        background-color: #ffebee;
        color: #c62828;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #ef9a9a;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Cached Data & Model Functions
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_cached_dataset(n_samples: int, seed: int) -> pd.DataFrame:
    """Generate and cache synthetic hydrology data."""
    sim = DataSimulator(random_seed=seed)
    return sim.generate_dataset(n_samples=n_samples)


def train_cached_engine(
    df: pd.DataFrame,
    features: List[str],
    continuous_target: str,
    categorical_target: str,
    test_size: float,
    random_state: int,
    n_neighbors: int,
    knn_weights: str,
    knn_metric: str,
) -> Tuple[HydroinformaticsEngine, RegressionMetrics, ClassificationMetrics]:
    """Train and return fitted Hydroinformatics Engine with evaluation metrics."""
    engine = HydroinformaticsEngine(
        features=features,
        continuous_target=continuous_target,
        categorical_target=categorical_target,
    )
    reg_metrics, clf_metrics = engine.fit_and_evaluate(
        df=df,
        test_size=test_size,
        random_state=random_state,
        n_neighbors=n_neighbors,
        knn_weights=knn_weights,
        knn_metric=knn_metric,
    )
    return engine, reg_metrics, clf_metrics


# -----------------------------------------------------------------------------
# Sidebar: Hydrological & Machine Learning Controls
# -----------------------------------------------------------------------------
st.sidebar.image(
    "https://img.icons8.com/fluency/96/water-pipe.png",
    width=64,
)
st.sidebar.title("Hydroinformatics Hub")
st.sidebar.caption("Predictive River Basin Intelligence System")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ 1. Hydrometric Telemetry")
n_samples = st.sidebar.slider(
    "Catchment Observation Samples",
    min_value=500,
    max_value=4000,
    value=1500,
    step=250,
    help="Number of synthetic daily station observation records.",
)
random_seed = st.sidebar.number_input(
    "Dataset Random Seed",
    min_value=1,
    max_value=9999,
    value=42,
    step=1,
)

st.sidebar.markdown("---")
st.sidebar.subheader("📐 2. ML Engine Hyperparameters")
test_split_ratio = st.sidebar.slider(
    "Train/Test Partition Ratio",
    min_value=0.10,
    max_value=0.40,
    value=0.20,
    step=0.05,
    help="Fraction of data reserved strictly for evaluation.",
)

continuous_target = st.sidebar.selectbox(
    "Continuous Target (Regression)",
    options=["Downstream_Streamflow_m3s", "Water_Quality_Index"],
    index=0,
    format_func=lambda x: "Downstream Streamflow (m³/s)" if x == "Downstream_Streamflow_m3s" else "Water Quality Index (WQI)",
)

k_neighbors = st.sidebar.slider(
    "KNN Neighbors (K)",
    min_value=1,
    max_value=25,
    value=5,
    step=2,
    help="Hyperparameter K: number of nearest neighbors for flood risk classification.",
)

knn_weights = st.sidebar.radio(
    "KNN Weighting Function",
    options=["uniform", "distance"],
    index=0,
    horizontal=True,
    help="'uniform': all neighbors contribute equally; 'distance': closer neighbors have greater weight.",
)

knn_metric = st.sidebar.selectbox(
    "KNN Distance Metric",
    options=["minkowski", "euclidean", "manhattan"],
    index=0,
)

# -----------------------------------------------------------------------------
# Data Simulation & Engine Training Pipeline
# -----------------------------------------------------------------------------
hydro_df = get_cached_dataset(n_samples=n_samples, seed=random_seed)

features = [
    "Daily_Rainfall_mm",
    "Soil_Moisture_Percentage",
    "Historical_River_Level_m",
    "Upstream_Flow_Rate",
]

engine, reg_metrics, clf_metrics = train_cached_engine(
    df=hydro_df,
    features=features,
    continuous_target=continuous_target,
    categorical_target="Flood_Risk_Level",
    test_size=test_split_ratio,
    random_state=random_seed,
    n_neighbors=k_neighbors,
    knn_weights=knn_weights,
    knn_metric=knn_metric,
)

# -----------------------------------------------------------------------------
# Main Header & High-Level KPIs
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">🌊 Predictive Hydroinformatics Tool</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Real-time Riverine Flood Classification (KNN) & Streamflow / Water Quality Forecasting (Linear Regression) '
    'with Strict Train-Test Pipeline Isolation.'
    '</div>',
    unsafe_allow_html=True,
)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Records", f"{len(hydro_df):,}")
with col2:
    st.metric("Train Partition", f"{len(engine.X_train):,}")
with col3:
    st.metric("Test Partition", f"{len(engine.X_test):,}")
with col4:
    st.metric("Regression R²", f"{reg_metrics.r2:.3f}")
with col5:
    st.metric("KNN Accuracy", f"{clf_metrics.accuracy * 100:.1f}%")

st.markdown("---")

# -----------------------------------------------------------------------------
# Interactive Tabs Organization
# -----------------------------------------------------------------------------
tab_scenario, tab_eval, tab_eda, tab_data = st.tabs([
    "🎯 Real-Time Scenario Forecast",
    "📊 Model Performance & Benchmarks",
    "🔍 Hydro-Environmental Analytics",
    "📄 Hydrologic Dataset Explorer"
])

# =============================================================================
# TAB 1: Real-Time Scenario Forecast & Inundation Assessment
# =============================================================================
with tab_scenario:
    st.subheader("Interactive Environmental State & Inundation Forecast")
    st.markdown(
        "Simulate arbitrary meteorological and hydrological conditions to assess downstream streamflow "
        "and instantaneous categorical flood risk."
    )

    # Preset scenarios for convenience
    preset_col, _ = st.columns([2, 3])
    with preset_col:
        preset = st.selectbox(
            "⚡ Quick Hydrologic Scenario Presets",
            options=[
                "Manual Custom Input",
                "Dry Summer Drought (Low Flow)",
                "Typical Autumn Rain (Moderate Flow)",
                "Severe Cloudburst / Extreme Flood Event",
                "High Saturated Spring Thaw",
            ],
            index=0,
        )

    # Set default values based on preset
    if preset == "Dry Summer Drought (Low Flow)":
        def_rain, def_soil, def_stage, def_up = 0.0, 22.0, 1.8, 65.0
    elif preset == "Typical Autumn Rain (Moderate Flow)":
        def_rain, def_soil, def_stage, def_up = 35.0, 58.0, 4.2, 280.0
    elif preset == "Severe Cloudburst / Extreme Flood Event":
        def_rain, def_soil, def_stage, def_up = 165.0, 92.0, 9.5, 980.0
    elif preset == "High Saturated Spring Thaw":
        def_rain, def_soil, def_stage, def_up = 85.0, 96.0, 7.8, 620.0
    else:
        def_rain, def_soil, def_stage, def_up = 65.0, 72.0, 5.5, 450.0

    st.markdown("#### Environmental Parameter Sliders")
    sc_col1, sc_col2 = st.columns(2)

    with sc_col1:
        input_rain = st.slider(
            "🌧️ Daily Rainfall (mm)",
            min_value=0.0,
            max_value=250.0,
            value=float(def_rain),
            step=1.0,
            help="Total 24-hour liquid precipitation accumulated in the catchment.",
        )
        input_soil = st.slider(
            "🌱 Soil Moisture Saturation (%)",
            min_value=10.0,
            max_value=100.0,
            value=float(def_soil),
            step=0.5,
            help="Volumetric moisture content of the topsoil layer.",
        )

    with sc_col2:
        input_stage = st.slider(
            "📏 Historical River Level (m)",
            min_value=1.0,
            max_value=12.0,
            value=float(def_stage),
            step=0.1,
            help="Baseline river gauge height prior to storm crest.",
        )
        input_upstream = st.slider(
            "🏞️ Upstream Flow Rate (m³/s)",
            min_value=25.0,
            max_value=1500.0,
            value=float(def_up),
            step=10.0,
            help="Headwater discharge rate flowing into the regional river basin.",
        )

    scenario_inputs = {
        "Daily_Rainfall_mm": input_rain,
        "Soil_Moisture_Percentage": input_soil,
        "Historical_River_Level_m": input_stage,
        "Upstream_Flow_Rate": input_upstream,
    }

    # Execute inference
    prediction = engine.predict_scenario(scenario_inputs)

    st.markdown("---")
    st.subheader("🔮 Predictive Model Output")

    out_col1, out_col2 = st.columns(2)

    with out_col1:
        st.markdown("### 1. Flood Inundation Risk (KNN Classifier)")
        risk_class = prediction.classification_prediction
        badge_class = f"risk-badge-{risk_class.lower()}"
        
        icon = "🟢" if risk_class == "Low" else ("🟡" if risk_class == "Medium" else "🔴")
        st.markdown(
            f'<div>Predicted Hazard Tier: <span class="{badge_class}">{icon} {risk_class.upper()} RISK</span></div>',
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**K-Nearest Neighbors Class Probabilities:**")
        prob_df = pd.DataFrame({
            "Risk Tier": list(prediction.classification_probabilities.keys()),
            "Probability": list(prediction.classification_probabilities.values()),
        })
        
        fig_prob = px.bar(
            prob_df,
            x="Risk Tier",
            y="Probability",
            color="Risk Tier",
            color_discrete_map={"Low": "#2e7d32", "Medium": "#f57f17", "High": "#c62828"},
            range_y=[0, 1.0],
            text_auto=".2%",
        )
        fig_prob.update_layout(height=240, margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
        st.plotly_chart(fig_prob, use_container_width=True)

        if risk_class == "Low":
            st.success("✅ **Hydrologic Advisory:** Discharge within standard channel bankfull capacity. Normal navigation & water abstraction.")
        elif risk_class == "Medium":
            st.warning("⚠️ **Hydrologic Watch:** River stage approaching bankfull capacity. Inundation risk in low-lying riparian floodplains.")
        else:
            st.error("🚨 **Hydrologic Warning:** Major overbank flood stage imminent! Immediate flood defence protocol recommended.")

    with out_col2:
        st.markdown(f"### 2. Continuous Target (Linear Regression)")
        val = prediction.regression_prediction

        if continuous_target == "Downstream_Streamflow_m3s":
            unit = "m³/s"
            st.metric(
                label="Forecasted Downstream Streamflow",
                value=f"{val:,.1f} {unit}",
                delta=f"{(val - input_upstream):+,.1f} m³/s vs Upstream",
            )
            # Physical context meter
            bankfull_capacity = 750.0
            capacity_pct = min(100.0, (val / bankfull_capacity) * 100.0)
            st.progress(
                int(capacity_pct),
                text=f"Hydraulic Channel Utilization: {capacity_pct:.1f}% of Bankfull Capacity ({bankfull_capacity} m³/s)"
            )
        else:
            unit = "Index Score (0 - 100)"
            st.metric(
                label="Forecasted Water Quality Index (WQI)",
                value=f"{val:.1f} / 100",
            )
            if val >= 80:
                st.success("🌿 **Water Quality Status:** Excellent to Good (Low Turbidity & Suspended Solids)")
            elif val >= 50:
                st.warning("🍂 **Water Quality Status:** Medium (Moderate Sediment Runoff)")
            else:
                st.error("🟤 **Water Quality Status:** Poor / Severely Degraded (High Surface Erosion & Solute Load)")

        # Display Regression Equation breakdown for this scenario
        st.markdown("**Scaled Equation Term Contributions:**")
        scaler = engine.regression_pipeline.named_steps["scaler"]
        reg = engine.regression_pipeline.named_steps["regressor"]
        
        # Calculate scaled contributions
        scaled_inputs = scaler.transform(pd.DataFrame([scenario_inputs])[engine.features])[0]
        term_contributions = scaled_inputs * reg.coef_

        contrib_df = pd.DataFrame({
            "Feature": [f.replace("_", " ") for f in engine.features],
            "Contribution Value": np.round(term_contributions, 2),
        })
        fig_contrib = px.bar(
            contrib_df,
            x="Contribution Value",
            y="Feature",
            orientation="h",
            color="Contribution Value",
            color_continuous_scale="Blues",
        )
        fig_contrib.update_layout(height=200, margin=dict(l=20, r=20, t=10, b=10))
        st.plotly_chart(fig_contrib, use_container_width=True)


# =============================================================================
# TAB 2: Model Performance & Benchmarks
# =============================================================================
with tab_eval:
    st.subheader("Model Validation & Hydroinformatics Performance Diagnostics")
    st.markdown(
        "Rigorous verification of the **Linear Regression** model (continuous estimation) and "
        "**K-Nearest Neighbors** model (categorical risk classification) on the isolated test partition."
    )

    # 1. Regression Evaluation Section
    st.markdown("### 📈 Continuous Target: Linear Regression Benchmark")
    reg_col1, reg_col2, reg_col3, reg_col4 = st.columns(4)
    with reg_col1:
        st.metric("Mean Squared Error (MSE)", f"{reg_metrics.mse:,.2f}")
    with reg_col2:
        st.metric("Root Mean Squared Error (RMSE)", f"{reg_metrics.rmse:,.2f}")
    with reg_col3:
        st.metric("R-Squared (R²)", f"{reg_metrics.r2:.4f}")
    with reg_col4:
        st.metric("Mean Absolute Error (MAE)", f"{reg_metrics.mae:,.2f}")

    # Scatter Plot: Actual vs Predicted
    y_test_pred = engine.regression_pipeline.predict(engine.X_test)
    eval_df = pd.DataFrame({
        "Actual": engine.y_reg_test.values,
        "Predicted": y_test_pred,
        "Residual": engine.y_reg_test.values - y_test_pred,
    })

    p_col1, p_col2 = st.columns(2)

    with p_col1:
        # Plotly Actual vs Predicted
        fig_scatter = go.Figure()
        # Identity line y = x
        min_val = min(eval_df["Actual"].min(), eval_df["Predicted"].min())
        max_val = max(eval_df["Actual"].max(), eval_df["Predicted"].max())
        
        fig_scatter.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="1:1 Perfect Fit (y = x)",
            line=dict(color="#d32f2f", dash="dash", width=2),
        ))
        fig_scatter.add_trace(go.Scatter(
            x=eval_df["Actual"],
            y=eval_df["Predicted"],
            mode="markers",
            name="Test Observations",
            marker=dict(color="#0288d1", size=6, opacity=0.75),
        ))
        fig_scatter.update_layout(
            title=f"Actual vs. Predicted ({continuous_target})",
            xaxis_title="Observed Ground Truth",
            yaxis_title="Linear Regression Prediction",
            height=400,
            margin=dict(l=30, r=30, t=40, b=30),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with p_col2:
        # Residuals Distribution
        fig_resid = px.histogram(
            eval_df,
            x="Residual",
            nbins=35,
            title="Residual Error Distribution (Actual - Predicted)",
            color_discrete_sequence=["#26a69a"],
            marginal="box",
        )
        fig_resid.add_vline(x=0, line_dash="dash", line_color="red")
        fig_resid.update_layout(
            xaxis_title="Residual Error",
            yaxis_title="Frequency Count",
            height=400,
            margin=dict(l=30, r=30, t=40, b=30),
        )
        st.plotly_chart(fig_resid, use_container_width=True)

    st.markdown("---")

    # 2. Classification Evaluation Section
    st.markdown("### 🎯 Categorical Hazard Target: KNN Classifier Benchmark")
    clf_col1, clf_col2 = st.columns([1, 1])

    with clf_col1:
        st.markdown(f"**Overall Classification Accuracy:** `{clf_metrics.accuracy * 100:.2f}%` (K={clf_metrics.n_neighbors})")
        st.markdown("**Detailed Classification Report:**")
        
        # Format classification report dictionary into structured DataFrame
        rep_dict = clf_metrics.report_dict
        report_rows = []
        for label in ["Low", "Medium", "High"]:
            if label in rep_dict:
                report_rows.append({
                    "Risk Tier": label,
                    "Precision": f"{rep_dict[label]['precision']:.3f}",
                    "Recall": f"{rep_dict[label]['recall']:.3f}",
                    "F1-Score": f"{rep_dict[label]['f1-score']:.3f}",
                    "Support": int(rep_dict[label]["support"]),
                })
        
        rep_df = pd.DataFrame(report_rows)
        st.dataframe(rep_df, hide_index=True, use_container_width=True)

        st.caption(
            f"Macro Avg F1: {rep_dict['macro avg']['f1-score']:.3f} | "
            f"Weighted Avg F1: {rep_dict['weighted avg']['f1-score']:.3f}"
        )

    with clf_col2:
        st.markdown("**Confusion Matrix Heatmap:**")
        cm_matrix = clf_metrics.confusion_mat
        cm_labels = clf_metrics.class_labels

        fig_cm, ax = plt.subplots(figsize=(4.5, 3.5))
        sns.heatmap(
            cm_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=cm_labels,
            yticklabels=cm_labels,
            cbar=False,
            ax=ax,
        )
        ax.set_xlabel("Predicted Flood Risk", fontsize=10, fontweight="bold")
        ax.set_ylabel("True Ground Truth", fontsize=10, fontweight="bold")
        ax.set_title(f"KNN Confusion Matrix (K={k_neighbors})", fontsize=11, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_cm)


# =============================================================================
# TAB 3: Exploratory Hydro-Data Analytics
# =============================================================================
with tab_eda:
    st.subheader("Catchment Hydrology Correlational Analysis")
    st.markdown(
        "Exploration of multi-variable interactions, showing how rainfall depth, antecedent soil moisture, "
        "and river levels influence flood hazard tiers."
    )

    eda_col1, eda_col2 = st.columns(2)

    with eda_col1:
        st.markdown("**Pearson Feature Correlation Heatmap:**")
        numeric_cols = [
            "Daily_Rainfall_mm",
            "Soil_Moisture_Percentage",
            "Historical_River_Level_m",
            "Upstream_Flow_Rate",
            "Downstream_Streamflow_m3s",
            "Water_Quality_Index",
        ]
        corr_matrix = hydro_df[numeric_cols].corr()

        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Correlation Matrix across Hydrologic Indicators",
        )
        fig_corr.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_corr, use_container_width=True)

    with eda_col2:
        st.markdown("**Hydrologic Phase Space (Rainfall vs. Soil Moisture):**")
        fig_phase = px.scatter(
            hydro_df,
            x="Daily_Rainfall_mm",
            y="Soil_Moisture_Percentage",
            color="Flood_Risk_Level",
            color_discrete_map={"Low": "#2e7d32", "Medium": "#f57f17", "High": "#c62828"},
            size="Downstream_Streamflow_m3s",
            hover_data=["Historical_River_Level_m", "Upstream_Flow_Rate"],
            title="Precipitation vs Antecedent Moisture (Sized by Flow)",
        )
        fig_phase.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_phase, use_container_width=True)


# =============================================================================
# TAB 4: Hydrologic Dataset Explorer
# =============================================================================
with tab_data:
    st.subheader("Hydrologic Records & Synthetic Data Telemetry")
    st.markdown(
        "Self-contained synthetic hydrological dataset generated using physically-consistent mass balance equations. "
        "Downloadable for external GIS/HEC-RAS hydrological modeling."
    )

    st.dataframe(hydro_df, use_container_width=True, height=350)

    d_col1, d_col2 = st.columns([1, 4])
    with d_col1:
        csv_bytes = hydro_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Dataset (CSV)",
            data=csv_bytes,
            file_name="hydroinformatics_synthetic_data.csv",
            mime="text/csv",
        )
    with d_col2:
        st.caption("Data generated with zero external network or file dependencies using DataSimulator.")

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #777; font-size: 0.85rem;'>"
    "Predictive Hydroinformatics Tool | Built with Scikit-Learn, Streamlit & Plotly | "
    "Principal Data Scientist & Hydroinformatics Engineering Specification"
    "</div>",
    unsafe_allow_html=True,
)

