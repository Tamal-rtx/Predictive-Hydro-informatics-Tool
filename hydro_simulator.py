"""Hydroinformatics Data Simulator Module.

Generates realistic, physically-consistent synthetic hydrological datasets
mimicking public hydrometric monitoring networks (e.g., USGS Water Data, NOAA, Copernicus).
"""

from typing import Tuple, List, Optional
import numpy as np
import pandas as pd


class DataSimulator:
    """Generates synthetic, physically-grounded hydro-environmental datasets for ML modeling.
    
    The generator incorporates hydrologic principles:
    - Runoff Generation: Non-linear soil moisture thresholding (Dunne/Hortonian runoff).
    - Hydrodynamic Streamflow Routing: Upstream tributary flow + direct catchment runoff.
    - Water Quality Dynamics: Runoff-induced suspended solids & dilution effects.
    - Hydraulic Inundation Risk: Stage-discharge capacity exceedance modeling.
    """

    FEATURE_COLUMNS: List[str] = [
        "Daily_Rainfall_mm",
        "Soil_Moisture_Percentage",
        "Historical_River_Level_m",
        "Upstream_Flow_Rate",
    ]

    CONTINUOUS_TARGET: str = "Downstream_Streamflow_m3s"
    SECONDARY_TARGET: str = "Water_Quality_Index"
    CATEGORICAL_TARGET: str = "Flood_Risk_Level"

    def __init__(self, random_seed: int = 42) -> None:
        """Initialize the simulator with a reproducible random seed.

        Args:
            random_seed (int): Random state for reproducibility.
        """
        self.random_seed: int = random_seed
        self.rng: np.random.RandomState = np.random.RandomState(random_seed)

    def generate_dataset(
        self,
        n_samples: int = 1500,
        noise_level: float = 0.05
    ) -> pd.DataFrame:
        """Simulate a comprehensive hydroinformatics dataframe.

        Args:
            n_samples (int): Number of daily hydrologic observation records to generate.
            noise_level (float): Gaussian noise dispersion factor to represent measurement variance.

        Returns:
            pd.DataFrame: Hydrological dataset with features and target variables.
        """
        self.rng = np.random.RandomState(self.random_seed)

        # 1. Daily Rainfall (mm): Right-skewed distribution (Gamma distribution)
        # Represents typical seasonal precipitation with episodic extreme events
        shape_k, scale_theta = 1.8, 18.0
        daily_rainfall = self.rng.gamma(shape=shape_k, scale=scale_theta, size=n_samples)
        # Add intermittent dry days (approx 35% dry spells typical in sub-humid catchments)
        dry_mask = self.rng.uniform(0, 1, size=n_samples) < 0.35
        daily_rainfall[dry_mask] = 0.0
        # Clip to realistic physical limits [0, 260 mm]
        daily_rainfall = np.clip(daily_rainfall, 0.0, 260.0)

        # 2. Soil Moisture Percentage (%): Beta distribution scaled to [12%, 98%]
        # Reflects antecedent soil saturation state and retention dynamics
        alpha_sm, beta_sm = 2.5, 2.2
        soil_moisture = self.rng.beta(alpha_sm, beta_sm, size=n_samples) * 86.0 + 12.0
        # Correlate soil moisture with recent rainfall
        soil_moisture = np.clip(
            soil_moisture + (daily_rainfall * 0.14) + self.rng.normal(0, 2.0, size=n_samples),
            10.0,
            100.0
        )

        # 3. Upstream Flow Rate (m^3/s): Lognormal headwater discharge distribution
        # Baseflow between 25 and 1400 m^3/s with extreme peak storm pulses
        log_mean, log_sigma = 5.2, 0.65
        upstream_flow = np.exp(self.rng.normal(log_mean, log_sigma, size=n_samples))
        upstream_flow = np.clip(upstream_flow, 25.0, 1500.0)

        # 4. Historical River Level (m): River stage height
        # Base stage between 1.2m and 11.5m, coupled with upstream discharge rating curve
        # Stage-discharge equation approximation: H = c * Q^(1/2) + noise
        river_level = 0.28 * np.sqrt(upstream_flow) + self.rng.normal(0, 0.35, size=n_samples)
        river_level = np.clip(river_level, 1.0, 12.0)

        # 5. Hydrological Physics Synthesis: Runoff Coefficient & Streamflow
        # Non-linear runoff generation: saturated soils convert a dramatically higher fraction to direct runoff
        # Logistic saturation threshold curve around 70% moisture
        moisture_fraction = soil_moisture / 100.0
        saturation_curve = 1.0 / (1.0 + np.exp(-10.0 * (moisture_fraction - 0.70)))
        runoff_coefficient = 0.10 + 0.75 * saturation_curve  # Range: [0.10, 0.85]

        # Effective surface runoff volume generated in catchment (m^3 equivalent index)
        effective_runoff = daily_rainfall * runoff_coefficient

        # Linear hydrological mass balance routing for Downstream Streamflow (m^3/s):
        # Q_down = 1.06 * Q_up + 4.85 * Effective_Runoff + 14.2 * River_Level + Baseflow + Noise
        flow_noise = self.rng.normal(0, 15.0 * (1.0 + noise_level), size=n_samples)
        downstream_flow = (
            1.06 * upstream_flow +
            4.85 * effective_runoff +
            14.2 * river_level +
            flow_noise
        )
        downstream_flow = np.clip(downstream_flow, 30.0, 2400.0)

        # 6. Water Quality Index (WQI): Scale 0 - 100
        # Low flow/clear water -> High WQI; Severe storm runoff washes sediments -> Drops WQI
        wqi_noise = self.rng.normal(0, 3.0, size=n_samples)
        wqi = (
            95.0
            - 0.22 * effective_runoff
            - 0.018 * upstream_flow
            - 1.4 * river_level
            + wqi_noise
        )
        wqi = np.clip(wqi, 15.0, 100.0)

        # 7. Flood Risk Classification: Physical Hydraulic Capacity Exceedance
        # In hydrologic warning systems (e.g. NOAA River Forecast Centers):
        # - Action/Normal stage: Low Risk (~60% of hydrologic regimes)
        # - Bankfull/Minor Flood stage: Medium Risk (~25% of hydrologic regimes)
        # - Moderate to Major Inundation: High Risk (~15% of hydrologic regimes)
        hydraulic_index = (
            0.60 * (downstream_flow / 450.0) +
            0.40 * (river_level / 5.5)
        )
        # Add stochastic environmental variance near critical thresholds
        hydraulic_index += self.rng.normal(0, 0.05, size=n_samples)

        flood_risk: List[str] = []
        for idx in hydraulic_index:
            if idx < 0.75:
                flood_risk.append("Low")
            elif idx < 1.15:
                flood_risk.append("Medium")
            else:
                flood_risk.append("High")

        df = pd.DataFrame({
            "Daily_Rainfall_mm": np.round(daily_rainfall, 2),
            "Soil_Moisture_Percentage": np.round(soil_moisture, 2),
            "Historical_River_Level_m": np.round(river_level, 2),
            "Upstream_Flow_Rate": np.round(upstream_flow, 2),
            "Catchment_Runoff_Coefficient": np.round(runoff_coefficient, 3),
            "Downstream_Streamflow_m3s": np.round(downstream_flow, 2),
            "Water_Quality_Index": np.round(wqi, 2),
            "Flood_Risk_Level": flood_risk,
        })

        return df


if __name__ == "__main__":
    simulator = DataSimulator(random_seed=42)
    sample_df = simulator.generate_dataset(n_samples=10)
    print("Synthetic Hydrological Data Sample (10 rows):")
    print(sample_df.to_string(index=False))
