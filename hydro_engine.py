"""Hydroinformatics Machine Learning Engine Module.

Implements Scikit-Learn Linear Regression and K-Nearest Neighbors (KNN)
with explicit feature scaling, strict train-test pipeline isolation,
and comprehensive hydrologic performance evaluation metrics.
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    classification_report,
    confusion_matrix,
    accuracy_score,
)

from hydro_simulator import DataSimulator


@dataclass
class RegressionMetrics:
    """Encapsulates continuous target regression evaluation metrics."""
    mse: float
    rmse: float
    r2: float
    mae: float
    target_name: str
    feature_names: List[str]
    coefficients: np.ndarray
    intercept: float


@dataclass
class ClassificationMetrics:
    """Encapsulates categorical flood risk classification evaluation metrics."""
    accuracy: float
    report_dict: Dict[str, Any]
    report_str: str
    confusion_mat: np.ndarray
    class_labels: List[str]
    target_name: str
    n_neighbors: int


@dataclass
class ScenarioPrediction:
    """Encapsulates real-time inference results for an environmental scenario."""
    regression_prediction: float
    regression_target_name: str
    classification_prediction: str
    classification_probabilities: Dict[str, float]
    input_features: Dict[str, float]


class HydroinformaticsEngine:
    """Object-Oriented Hydroinformatics ML Pipeline.
    
    Houses both continuous regression (Linear Regression) and
    categorical risk classification (KNN) workflows with rigorous
    pipeline isolation and feature normalization.
    """

    CLASS_LABELS: List[str] = ["Low", "Medium", "High"]

    def __init__(
        self,
        features: Optional[List[str]] = None,
        continuous_target: str = "Downstream_Streamflow_m3s",
        categorical_target: str = "Flood_Risk_Level",
    ) -> None:
        """Initialize the Hydroinformatics Engine.

        Args:
            features (Optional[List[str]]): List of input feature names.
            continuous_target (str): Name of continuous regression target column.
            categorical_target (str): Name of categorical flood classification column.
        """
        self.features: List[str] = features or [
            "Daily_Rainfall_mm",
            "Soil_Moisture_Percentage",
            "Historical_River_Level_m",
            "Upstream_Flow_Rate",
        ]
        self.continuous_target: str = continuous_target
        self.categorical_target: str = categorical_target

        # Pipelines
        self.regression_pipeline: Optional[Pipeline] = None
        self.classification_pipeline: Optional[Pipeline] = None

        # Train/Test Data Cache for diagnostics & plotting
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_reg_train: Optional[pd.Series] = None
        self.y_reg_test: Optional[pd.Series] = None
        self.y_clf_train: Optional[pd.Series] = None
        self.y_clf_test: Optional[pd.Series] = None

        # Evaluated Metrics
        self.regression_metrics: Optional[RegressionMetrics] = None
        self.classification_metrics: Optional[ClassificationMetrics] = None

    def partition_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.20,
        random_state: int = 42
    ) -> None:
        """Partition hydrologic data into training and evaluation sets.
        
        Preserves pipeline isolation and stratification for classification risk.

        Args:
            df (pd.DataFrame): Hydrologic dataframe.
            test_size (float): Proportion of dataset to hold out for evaluation.
            random_state (int): Reproducibility seed for data split.
        """
        X = df[self.features]
        y_reg = df[self.continuous_target]
        y_clf = df[self.categorical_target]

        # Stratified train-test split based on classification labels
        (
            self.X_train,
            self.X_test,
            self.y_reg_train,
            self.y_reg_test,
            self.y_clf_train,
            self.y_clf_test
        ) = train_test_split(
            X,
            y_reg,
            y_clf,
            test_size=test_size,
            random_state=random_state,
            stratify=y_clf
        )

    def train_regression_model(self) -> Pipeline:
        """Construct and fit the Linear Regression pipeline with StandardScaler.

        Returns:
            Pipeline: Trained Scikit-Learn Pipeline.
        """
        if self.X_train is None or self.y_reg_train is None:
            raise ValueError("Training data must be partitioned before fitting regression model.")

        # Scaler fit exclusively on training data to prevent lookahead bias / data leakage
        self.regression_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression())
        ])

        self.regression_pipeline.fit(self.X_train, self.y_reg_train)
        return self.regression_pipeline

    def train_classification_model(
        self,
        n_neighbors: int = 5,
        weights: str = "uniform",
        metric: str = "minkowski"
    ) -> Pipeline:
        """Construct and fit the K-Nearest Neighbors Classifier pipeline with StandardScaler.

        Args:
            n_neighbors (int): Number of nearest neighbors (K).
            weights (str): Weight function used in prediction ('uniform' or 'distance').
            metric (str): Distance metric to use for tree ('minkowski', 'euclidean', 'manhattan').

        Returns:
            Pipeline: Trained Scikit-Learn KNN Pipeline.
        """
        if self.X_train is None or self.y_clf_train is None:
            raise ValueError("Training data must be partitioned before fitting classifier model.")

        self.classification_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights=weights,
                metric=metric
            ))
        ])

        self.classification_pipeline.fit(self.X_train, self.y_clf_train)
        return self.classification_pipeline

    def evaluate_regression(self) -> RegressionMetrics:
        """Evaluate the regression pipeline against test partition.

        Returns:
            RegressionMetrics: Dataclass containing MSE, RMSE, R^2, and MAE.
        """
        if self.regression_pipeline is None or self.X_test is None or self.y_reg_test is None:
            raise ValueError("Regression pipeline has not been trained or test partition missing.")

        y_pred = self.regression_pipeline.predict(self.X_test)

        mse = float(mean_squared_error(self.y_reg_test, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(self.y_reg_test, y_pred))
        mae = float(mean_absolute_error(self.y_reg_test, y_pred))

        regressor: LinearRegression = self.regression_pipeline.named_steps["regressor"]
        coefficients = regressor.coef_
        intercept = float(regressor.intercept_)

        self.regression_metrics = RegressionMetrics(
            mse=mse,
            rmse=rmse,
            r2=r2,
            mae=mae,
            target_name=self.continuous_target,
            feature_names=self.features,
            coefficients=coefficients,
            intercept=intercept
        )
        return self.regression_metrics

    def evaluate_classification(self) -> ClassificationMetrics:
        """Evaluate the KNN classification pipeline against test partition.

        Returns:
            ClassificationMetrics: Dataclass containing Accuracy, Report, and Confusion Matrix.
        """
        if self.classification_pipeline is None or self.X_test is None or self.y_clf_test is None:
            raise ValueError("Classification pipeline has not been trained or test partition missing.")

        y_pred = self.classification_pipeline.predict(self.X_test)

        acc = float(accuracy_score(self.y_clf_test, y_pred))
        report_dict = classification_report(
            self.y_clf_test,
            y_pred,
            labels=self.CLASS_LABELS,
            output_dict=True,
            zero_division=0
        )
        report_str = classification_report(
            self.y_clf_test,
            y_pred,
            labels=self.CLASS_LABELS,
            zero_division=0
        )
        c_matrix = confusion_matrix(
            self.y_clf_test,
            y_pred,
            labels=self.CLASS_LABELS
        )

        knn_model: KNeighborsClassifier = self.classification_pipeline.named_steps["classifier"]

        self.classification_metrics = ClassificationMetrics(
            accuracy=acc,
            report_dict=report_dict,
            report_str=report_str,
            confusion_mat=c_matrix,
            class_labels=self.CLASS_LABELS,
            target_name=self.categorical_target,
            n_neighbors=knn_model.n_neighbors
        )
        return self.classification_metrics

    def fit_and_evaluate(
        self,
        df: pd.DataFrame,
        test_size: float = 0.20,
        random_state: int = 42,
        n_neighbors: int = 5,
        knn_weights: str = "uniform",
        knn_metric: str = "minkowski"
    ) -> Tuple[RegressionMetrics, ClassificationMetrics]:
        """Execute end-to-end data partitioning, training, and evaluation.

        Args:
            df (pd.DataFrame): Hydrological dataset.
            test_size (float): Holdout validation ratio.
            random_state (int): Split seed.
            n_neighbors (int): Hyperparameter K for KNN.
            knn_weights (str): Weight parameter for KNN.
            knn_metric (str): Distance metric.

        Returns:
            Tuple[RegressionMetrics, ClassificationMetrics]: Computed benchmark metrics.
        """
        self.partition_data(df, test_size=test_size, random_state=random_state)
        self.train_regression_model()
        self.train_classification_model(
            n_neighbors=n_neighbors,
            weights=knn_weights,
            metric=knn_metric
        )
        reg_metrics = self.evaluate_regression()
        clf_metrics = self.evaluate_classification()
        return reg_metrics, clf_metrics

    def predict_scenario(self, features_dict: Dict[str, float]) -> ScenarioPrediction:
        """Perform real-time continuous and categorical inference for a given environmental state.

        Args:
            features_dict (Dict[str, float]): Dictionary mapping feature names to numerical values.

        Returns:
            ScenarioPrediction: Unified prediction record.
        """
        if self.regression_pipeline is None or self.classification_pipeline is None:
            raise RuntimeError("Models must be trained before predicting scenarios.")

        # Reconstruct DataFrame matching training schema
        input_row = pd.DataFrame([features_dict])[self.features]

        # 1. Regression continuous forecast
        reg_pred = float(self.regression_pipeline.predict(input_row)[0])

        # 2. Classification categorical prediction
        clf_pred = str(self.classification_pipeline.predict(input_row)[0])

        # 3. Probabilities for each risk category
        prob_array = self.classification_pipeline.predict_proba(input_row)[0]
        knn_classes = list(self.classification_pipeline.named_steps["classifier"].classes_)
        prob_dict = {cls: float(prob) for cls, prob in zip(knn_classes, prob_array)}

        # Ensure all standard labels are represented
        for label in self.CLASS_LABELS:
            if label not in prob_dict:
                prob_dict[label] = 0.0

        return ScenarioPrediction(
            regression_prediction=reg_pred,
            regression_target_name=self.continuous_target,
            classification_prediction=clf_pred,
            classification_probabilities=prob_dict,
            input_features=features_dict
        )

    def print_evaluation_summary(self) -> None:
        """Output formatted performance benchmarks to standard output."""
        print("=" * 80)
        print("PREDICTIVE HYDROINFORMATICS ENGINE: MODEL VERIFICATION REPORT")
        print("=" * 80)

        if self.regression_metrics:
            rm = self.regression_metrics
            print(f"\n--- REGRESSION BENCHMARK: Linear Regression [{rm.target_name}] ---")
            print(f"  Mean Squared Error (MSE)      : {rm.mse:12.4f}")
            print(f"  Root Mean Squared Error (RMSE): {rm.rmse:12.4f}")
            print(f"  R-squared (R^2) Score         : {rm.r2:12.4f}")
            print(f"  Mean Absolute Error (MAE)     : {rm.mae:12.4f}")
            print("\n  Fitted Scaled Regression Formula:")
            formula_terms = [f"({coef:+.3f} * {feat})" for feat, coef in zip(rm.feature_names, rm.coefficients)]
            print(f"  {rm.target_name} = {rm.intercept:.3f} + " + " + ".join(formula_terms))

        if self.classification_metrics:
            cm = self.classification_metrics
            print(f"\n--- CLASSIFICATION BENCHMARK: K-Nearest Neighbors (K={cm.n_neighbors}) ---")
            print(f"  Overall Validation Accuracy   : {cm.accuracy * 100:.2f}%\n")
            print("  Classification Report:")
            print(cm.report_str)
            print("  Confusion Matrix:")
            print(f"  Classes Order: {cm.class_labels}")
            header = "          " + " ".join([f"{lbl:>8}" for lbl in cm.class_labels])
            print(header)
            for i, row in enumerate(cm.confusion_mat):
                row_str = f"  {cm.class_labels[i]:>8} " + " ".join([f"{val:8d}" for val in row])
                print(row_str)

        print("=" * 80)


if __name__ == "__main__":
    # Demonstration CLI execution of the end-to-end hydroinformatics system
    print("[1/3] Generating synthetic hydrological observations via DataSimulator...")
    sim = DataSimulator(random_seed=42)
    hydro_df = sim.generate_dataset(n_samples=1500)

    print(f"[2/3] Initializing ML Engine for {sim.CONTINUOUS_TARGET} and {sim.CATEGORICAL_TARGET}...")
    engine = HydroinformaticsEngine(
        features=sim.FEATURE_COLUMNS,
        continuous_target=sim.CONTINUOUS_TARGET,
        categorical_target=sim.CATEGORICAL_TARGET,
    )

    print("[3/3] Training pipelines (StandardScaler + LinearRegression & KNN) and evaluating...")
    engine.fit_and_evaluate(
        df=hydro_df,
        test_size=0.20,
        random_state=42,
        n_neighbors=5,
        knn_weights="distance"
    )

    engine.print_evaluation_summary()

    # Sample scenario prediction
    test_scenario = {
        "Daily_Rainfall_mm": 135.0,
        "Soil_Moisture_Percentage": 88.5,
        "Historical_River_Level_m": 8.4,
        "Upstream_Flow_Rate": 850.0,
    }
    prediction = engine.predict_scenario(test_scenario)
    print("\n--- INFERENCE TEST SCENARIO ---")
    print(f"Scenario Inputs : {test_scenario}")
    print(f"Predicted Flow  : {prediction.regression_prediction:.2f} m^3/s")
    print(f"Flood Risk Tier : {prediction.classification_prediction.upper()} RISK")
    print(f"Risk Probabilities: {prediction.classification_probabilities}")

