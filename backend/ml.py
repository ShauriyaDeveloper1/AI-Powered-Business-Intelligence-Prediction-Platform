import json
import os
import re
import time
from difflib import get_close_matches
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_models")
CLASSIFIER_MODEL_PATH = os.path.join(MODELS_DIR, "classifier_bundle.pkl")
REGRESSOR_MODEL_PATH = os.path.join(MODELS_DIR, "regressor_bundle.pkl")
SEGMENTATION_MODEL_PATH = os.path.join(MODELS_DIR, "segmentation_bundle.pkl")
MISSING_VALUE_MARKERS = {"", " ", "na", "n/a", "null", "none", "nan", "-", "?"}


@dataclass
class TrainResult:
    task_type: str
    target_column: str
    total_records: int
    processed_records: int
    preview: List[Dict[str, Any]]
    dataset_rows_sample: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    cluster_plot: Dict[str, Any]
    cluster_summary: List[Dict[str, Any]]
    feature_importance: List[Dict[str, Any]]
    insight_text: str
    column_mapping: Dict[str, str]
    cleaning_report: Dict[str, Any]
    training_time_seconds: float = 0.0


_last_context: Dict[str, Any] = {}

CLASSIFICATION_TARGET_PREFERENCES = [
    "target",
    "label",
    "class",
    "outcome",
    "status",
    "churn_label",
    "churn_value",
    "churn",
    "exited",
    "is_churned",
    "defaulted",
    "fraud",
    "response",
    "converted",
]


def _normalize_name(name: Any) -> str:
    token = str(name).strip().lower()
    token = re.sub(r"[^a-z0-9]+", "_", token)
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "column"


def _normalize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, str]]:
    mapping: Dict[str, str] = {}
    used: Dict[str, int] = {}
    normalized_names: List[str] = []

    for original in df.columns:
        base = _normalize_name(original)
        seen = used.get(base, 0)
        used[base] = seen + 1
        normalized = base if seen == 0 else f"{base}_{seen + 1}"
        mapping[str(original)] = normalized
        normalized_names.append(normalized)

    normalized_df = df.copy()
    normalized_df.columns = normalized_names
    return normalized_df, mapping


def _resolve_column_name(requested: str, available_columns: List[str]) -> Optional[str]:
    if requested in available_columns:
        return requested

    normalized_to_actual = {_normalize_name(column): column for column in available_columns}
    normalized_requested = _normalize_name(requested)
    if normalized_requested in normalized_to_actual:
        return normalized_to_actual[normalized_requested]

    close = get_close_matches(normalized_requested, list(normalized_to_actual.keys()), n=1, cutoff=0.7)
    if close:
        return normalized_to_actual[close[0]]
    return None


def _infer_default_target_column(df: pd.DataFrame, task_type: str) -> str:
    columns = df.columns.tolist()

    if task_type == "classification":
        normalized_to_actual = {_normalize_name(column): column for column in columns}
        for preferred in CLASSIFICATION_TARGET_PREFERENCES:
            resolved = normalized_to_actual.get(_normalize_name(preferred))
            if resolved:
                return resolved

        candidate_columns = []
        for column in columns:
            series = df[column]
            if series.isna().all():
                continue

            distinct_values = int(series.nunique(dropna=True))
            if distinct_values < 2 or distinct_values > 20:
                continue

            normalized_name = _normalize_name(column)
            if normalized_name.endswith("id") or normalized_name.endswith("_id"):
                continue

            candidate_columns.append((distinct_values, column))

        if candidate_columns:
            candidate_columns.sort(key=lambda item: (item[0], -columns.index(item[1])))
            return candidate_columns[0][1]

    return columns[-1]


def _clean_object_values(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    object_columns = cleaned.select_dtypes(include=["object"]).columns.tolist()
    for column in object_columns:
        values = cleaned[column].astype(str).str.strip()
        values = values.replace({marker: np.nan for marker in MISSING_VALUE_MARKERS})
        cleaned[column] = values
    return cleaned


def _coerce_numeric_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    coerced = df.copy()
    object_columns = coerced.select_dtypes(include=["object"]).columns.tolist()

    for column in object_columns:
        non_null_values = coerced[column].dropna()
        if non_null_values.empty:
            continue

        normalized = non_null_values.astype(str).str.replace(",", "", regex=False)
        numeric_candidate = pd.to_numeric(normalized, errors="coerce")
        converted_ratio = float(numeric_candidate.notna().mean())
        if converted_ratio < 0.85:
            continue

        coerced[column] = pd.to_numeric(coerced[column].astype(str).str.replace(",", "", regex=False), errors="coerce")

    return coerced


def _prepare_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_cols = [column for column in X.columns if column not in categorical_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    clean_df = _clean_object_values(df)
    clean_df = _coerce_numeric_like_columns(clean_df)
    clean_df = clean_df.replace([np.inf, -np.inf], np.nan)
    clean_df = clean_df.drop_duplicates()
    return clean_df


def _get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    feature_names = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        if not cols:
            continue
        if hasattr(transformer, "named_steps") and "encoder" in transformer.named_steps:
            encoder = transformer.named_steps["encoder"]
            try:
                encoded = encoder.get_feature_names_out(cols)
                feature_names.extend(encoded.tolist())
            except Exception:
                feature_names.extend(cols)
        else:
            feature_names.extend(cols)
    return feature_names


def _select_optimal_k(X_processed: Any, min_k: int = 2, max_k: int = 7) -> int:
    dense_matrix = X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed
    sample_size = len(dense_matrix)
    if sample_size < 3:
        return 2

    max_k = min(max_k, max(2, sample_size - 1))
    inertias = []
    k_values = list(range(min_k, max_k + 1))

    for candidate_k in k_values:
        model = KMeans(n_clusters=candidate_k, random_state=42, n_init=10)
        model.fit(dense_matrix)
        inertias.append(model.inertia_)

    if len(inertias) <= 2:
        return k_values[-1]

    drops = [inertias[idx - 1] - inertias[idx] for idx in range(1, len(inertias))]
    acceleration = [drops[idx - 1] - drops[idx] for idx in range(1, len(drops))]
    elbow_index = int(np.argmax(acceleration)) + 1
    return int(k_values[elbow_index])


def _cluster_payload(X_processed: Any, n_clusters: Optional[int] = None):
    if n_clusters is None:
        n_clusters = _select_optimal_k(X_processed)

    cluster_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = cluster_model.fit_predict(X_processed)

    dense_matrix = X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed
    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(dense_matrix)

    points = [
        {
            "x": float(reduced[idx][0]),
            "y": float(reduced[idx][1]),
            "cluster": int(cluster_labels[idx]),
        }
        for idx in range(len(cluster_labels))
    ]

    summary = []
    cluster_names = ["High value", "Risk", "Average", "Growth"]
    for cluster_id in range(n_clusters):
        count = int(np.sum(cluster_labels == cluster_id))
        summary.append(
            {
                "cluster": cluster_id,
                "label": f"Cluster {cluster_id + 1} - {cluster_names[cluster_id % len(cluster_names)]}",
                "count": count,
            }
        )

    return {"points": points}, summary


def _generate_insight(top_features: List[Dict[str, Any]], task_type: str) -> str:
    if not top_features:
        return "No dominant feature pattern detected yet."

    lead = top_features[0]
    follow = top_features[1] if len(top_features) > 1 else None
    if task_type == "classification":
        if follow:
            return (
                f"Primary churn influence is '{lead['feature']}' ({lead['importance']:.3f}), "
                f"followed by '{follow['feature']}' ({follow['importance']:.3f}). "
                "Prioritize retention interventions around these factors."
            )
        return (
            f"Primary churn influence is '{lead['feature']}' ({lead['importance']:.3f}). "
            "Use this as a first decision lever for retention actions."
        )

    if task_type == "segmentation":
        if follow:
            return (
                f"Segments are primarily separated by '{lead['feature']}' ({lead['importance']:.3f}) "
                f"and '{follow['feature']}' ({follow['importance']:.3f}). "
                "Use these drivers for segment-wise engagement strategies."
            )
        return (
            f"Segments are primarily separated by '{lead['feature']}' ({lead['importance']:.3f}). "
            "Use this driver as the first lever for segment targeting."
        )

    if follow:
        return (
            f"Strongest regression driver is '{lead['feature']}' ({lead['importance']:.3f}), "
            f"then '{follow['feature']}' ({follow['importance']:.3f}). "
            "Optimize these variables to improve forecast outcomes."
        )
    return (
        f"Strongest regression driver is '{lead['feature']}' ({lead['importance']:.3f}). "
        "Monitor this variable closely for forecast optimization."
    )


def _classification_rf_candidates(is_imbalanced: bool) -> List[Dict[str, Any]]:
    return [
        {
            "n_estimators": 200,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "class_weight": "balanced_subsample" if is_imbalanced else None,
            "random_state": 42,
        },
        {
            "n_estimators": 350,
            "max_depth": None,
            "min_samples_split": 4,
            "min_samples_leaf": 1,
            "class_weight": "balanced_subsample" if is_imbalanced else None,
            "random_state": 42,
        },
        {
            "n_estimators": 300,
            "max_depth": 16,
            "min_samples_split": 6,
            "min_samples_leaf": 2,
            "class_weight": "balanced_subsample" if is_imbalanced else None,
            "random_state": 42,
        },
    ]


def _regression_rf_candidates() -> List[Dict[str, Any]]:
    return [
        {
            "n_estimators": 250,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "random_state": 42,
        },
        {
            "n_estimators": 400,
            "max_depth": None,
            "min_samples_split": 4,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "random_state": 42,
        },
        {
            "n_estimators": 300,
            "max_depth": 18,
            "min_samples_split": 6,
            "min_samples_leaf": 2,
            "max_features": 1.0,
            "random_state": 42,
        },
    ]


def train_models(file_path: str, task_type: str = "classification", target_column: str | None = None) -> TrainResult:
    _train_start = time.time()
    os.makedirs(MODELS_DIR, exist_ok=True)

    raw_df = pd.read_csv(file_path, keep_default_na=True, na_values=sorted(MISSING_VALUE_MARKERS))
    df, column_mapping = _normalize_columns(raw_df)
    total_records = len(df)

    if total_records == 0:
        raise ValueError("Dataset is empty")

    if task_type == "segmentation":
        clean_df = _clean_dataframe(df)
        duplicates_removed = int(total_records - len(clean_df))
        if clean_df.empty:
            raise ValueError("Dataset became empty after preprocessing")

        all_null_cols = [column for column in clean_df.columns if clean_df[column].isna().all()]
        dropped_all_missing = all_null_cols[:]
        if dropped_all_missing:
            clean_df = clean_df.drop(columns=dropped_all_missing)

        if clean_df.empty or not clean_df.columns.tolist():
            raise ValueError("Dataset has no usable feature columns after cleaning")

        numeric_cols = clean_df.select_dtypes(include=["number", "bool"]).columns.tolist()
        if not numeric_cols:
            raise ValueError("Segmentation requires numeric feature columns")

        X = clean_df[numeric_cols].copy()
        preprocessor = _prepare_preprocessor(X)
        X_processed = preprocessor.fit_transform(X)

        cluster_plot, cluster_summary = _cluster_payload(X_processed)

        # Use normalized variance contribution as an unsupervised proxy for feature influence.
        variance_series = X.var(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        variance_sum = float(variance_series.sum())
        if variance_sum <= 0:
            top_features = [{"feature": column, "importance": 0.0} for column in numeric_cols[:10]]
        else:
            top_features = sorted(
                [
                    {"feature": str(column), "importance": float(variance_series[column] / variance_sum)}
                    for column in numeric_cols
                ],
                key=lambda item: item["importance"],
                reverse=True,
            )[:10]

        insight_text = _generate_insight(top_features, "segmentation")

        cleaning_report = {
            "original_columns": raw_df.columns.astype(str).tolist(),
            "normalized_columns": df.columns.astype(str).tolist(),
            "target_column_requested": None,
            "target_column_resolved": None,
            "duplicates_removed": duplicates_removed,
            "rows_removed_missing_target": 0,
            "rows_removed_invalid_numeric_target": 0,
            "rows_retained_for_training": int(len(clean_df)),
            "dropped_all_missing_columns": dropped_all_missing,
            "selected_numeric_columns": numeric_cols,
        }

        metrics = {
            "segmentation": {
                "segments": len(cluster_summary),
                "records_clustered": int(len(clean_df)),
            },
            "model_comparison": {
                "winner": "KMeans",
                "based_on": "Unsupervised cluster partitioning",
            },
            "dashboard": {
                "accuracy": None,
                "churn_rate": None,
            },
        }

        bundle = {
            "task_type": "segmentation",
            "target_column": None,
            "pipeline": None,
            "expected_features": numeric_cols,
            "column_mapping": column_mapping,
            "feature_importance": top_features,
            "insight_text": insight_text,
            "cleaning_report": cleaning_report,
            "metrics": metrics,
        }
        joblib.dump(bundle, SEGMENTATION_MODEL_PATH)

        _last_context.clear()
        _last_context.update(bundle)
        _last_context["cluster_plot"] = cluster_plot
        _last_context["cluster_summary"] = cluster_summary

        return TrainResult(
            task_type="segmentation",
            target_column="N/A",
            total_records=total_records,
            processed_records=len(clean_df),
            preview=clean_df.head(10).to_dict(orient="records"),
            dataset_rows_sample=clean_df.head(1000).to_dict(orient="records"),
            metrics=metrics,
            cluster_plot=cluster_plot,
            cluster_summary=cluster_summary,
            feature_importance=top_features,
            insight_text=insight_text,
            column_mapping=column_mapping,
            cleaning_report=cleaning_report,
            training_time_seconds=round(time.time() - _train_start, 2),
        )

    requested_target_column = target_column

    if target_column is None:
        target_column = _infer_default_target_column(df, task_type)
    else:
        resolved_target = _resolve_column_name(target_column, df.columns.tolist())
        if not resolved_target:
            raise ValueError(
                f"Target column '{target_column}' not found. Available columns: {', '.join(df.columns.tolist())}"
            )
        target_column = resolved_target

    clean_df = _clean_dataframe(df)
    duplicates_removed = int(total_records - len(clean_df))
    if clean_df.empty:
        raise ValueError("Dataset became empty after preprocessing")

    rows_removed_missing_target = int(clean_df[target_column].isna().sum())
    if rows_removed_missing_target:
        clean_df = clean_df.loc[clean_df[target_column].notna()].copy()

    rows_removed_invalid_numeric_target = 0
    if task_type == "regression":
        numeric_target = pd.to_numeric(clean_df[target_column], errors="coerce")
        rows_removed_invalid_numeric_target = int(numeric_target.isna().sum())
        if rows_removed_invalid_numeric_target:
            clean_df = clean_df.loc[numeric_target.notna()].copy()
            numeric_target = numeric_target.loc[numeric_target.notna()]
        clean_df[target_column] = numeric_target.astype(float)

    if clean_df.empty:
        raise ValueError("Dataset became empty after removing rows with invalid target values")

    all_null_cols = [column for column in clean_df.columns if clean_df[column].isna().all()]
    if target_column in all_null_cols:
        raise ValueError(f"Target column '{target_column}' has only missing values")

    dropped_all_missing = [column for column in all_null_cols if column != target_column]
    if dropped_all_missing:
        clean_df = clean_df.drop(columns=dropped_all_missing)

    X = clean_df.drop(columns=[target_column])
    y = clean_df[target_column]

    if X.empty or not X.columns.tolist():
        raise ValueError("Dataset has no usable feature columns after cleaning")
    if len(clean_df) < 2:
        raise ValueError("Need at least 2 valid rows after cleaning to train a model")

    cleaning_report = {
        "original_columns": raw_df.columns.astype(str).tolist(),
        "normalized_columns": df.columns.astype(str).tolist(),
        "target_column_requested": requested_target_column,
        "target_column_resolved": target_column,
        "duplicates_removed": duplicates_removed,
        "rows_removed_missing_target": rows_removed_missing_target,
        "rows_removed_invalid_numeric_target": rows_removed_invalid_numeric_target,
        "rows_retained_for_training": int(len(clean_df)),
        "dropped_all_missing_columns": dropped_all_missing,
    }

    preprocessor = _prepare_preprocessor(X)

    test_size = 0.2 if len(clean_df) >= 10 else 0.3
    random_state = 42

    stratify_target = None
    if task_type == "classification" and y.nunique() > 1:
        class_counts = y.value_counts(dropna=False)
        if int(class_counts.min()) >= 2:
            stratify_target = y

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_target,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )

    if task_type == "classification":
        if y.nunique(dropna=True) < 2:
            raise ValueError("Need at least 2 target classes after cleaning to train a classification model")

        is_imbalanced = False
        if y.nunique() > 1:
            class_ratio = y.value_counts(normalize=True, dropna=False)
            is_imbalanced = float(class_ratio.max()) > 0.75

        baseline_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2500,
                        class_weight="balanced" if is_imbalanced else None,
                    ),
                ),
            ]
        )

        baseline_pipeline.fit(X_train, y_train)
        baseline_pred = baseline_pipeline.predict(X_test)

        baseline_metrics = {
            "accuracy": float(accuracy_score(y_test, baseline_pred)),
            "precision": float(precision_score(y_test, baseline_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, baseline_pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_test, baseline_pred, average="weighted", zero_division=0)),
        }

        best_rf_pipeline = None
        best_rf_metrics = None
        best_rf_params = None
        best_rf_score = float("-inf")

        for candidate in _classification_rf_candidates(is_imbalanced):
            candidate_pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", RandomForestClassifier(**candidate)),
                ]
            )
            candidate_pipeline.fit(X_train, y_train)
            candidate_pred = candidate_pipeline.predict(X_test)
            candidate_metrics = {
                "accuracy": float(accuracy_score(y_test, candidate_pred)),
                "precision": float(precision_score(y_test, candidate_pred, average="weighted", zero_division=0)),
                "recall": float(recall_score(y_test, candidate_pred, average="weighted", zero_division=0)),
                "f1": float(f1_score(y_test, candidate_pred, average="weighted", zero_division=0)),
            }

            score = candidate_metrics["f1"] + (candidate_metrics["recall"] * 0.01)
            if score > best_rf_score:
                best_rf_score = score
                best_rf_pipeline = candidate_pipeline
                best_rf_metrics = candidate_metrics
                best_rf_params = candidate

        final_pipeline = best_rf_pipeline
        final_metrics = best_rf_metrics or {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

        transformed = final_pipeline.named_steps["preprocessor"].transform(X)
        cluster_plot, cluster_summary = _cluster_payload(transformed)

        trained_preprocessor = final_pipeline.named_steps["preprocessor"]
        rf_model = final_pipeline.named_steps["model"]
        feature_names = _get_feature_names(trained_preprocessor)
        importances = rf_model.feature_importances_

        top_features = sorted(
            [
                {"feature": feature_names[idx], "importance": float(importances[idx])}
                for idx in range(len(feature_names))
            ],
            key=lambda item: item["importance"],
            reverse=True,
        )[:10]

        insight_text = _generate_insight(top_features, "classification")

        churn_rate = float((clean_df[target_column].astype(str).str.lower().isin(["1", "yes", "true", "churn", "churned"]).mean()))

        metrics = {
            "classification": {
                "logistic_regression": baseline_metrics,
                "random_forest": final_metrics,
            },
            "model_comparison": {
                "winner": "Random Forest" if final_metrics["f1"] >= baseline_metrics["f1"] else "Logistic Regression",
                "based_on": "F1-score",
                "selected_random_forest_params": best_rf_params,
            },
            "dashboard": {
                "accuracy": final_metrics["accuracy"],
                "churn_rate": churn_rate,
            },
        }

        bundle = {
            "task_type": "classification",
            "target_column": target_column,
            "pipeline": final_pipeline,
            "expected_features": X.columns.tolist(),
            "column_mapping": column_mapping,
            "feature_importance": top_features,
            "insight_text": insight_text,
            "cleaning_report": cleaning_report,
            "metrics": metrics,
        }
        joblib.dump(bundle, CLASSIFIER_MODEL_PATH)

        _last_context.clear()
        _last_context.update(bundle)
        _last_context["cluster_plot"] = cluster_plot
        _last_context["cluster_summary"] = cluster_summary

        return TrainResult(
            task_type="classification",
            target_column=target_column,
            total_records=total_records,
            processed_records=len(clean_df),
            preview=clean_df.head(10).to_dict(orient="records"),
            dataset_rows_sample=clean_df.head(1000).to_dict(orient="records"),
            metrics=metrics,
            cluster_plot=cluster_plot,
            cluster_summary=cluster_summary,
            feature_importance=top_features,
            insight_text=insight_text,
            column_mapping=column_mapping,
            cleaning_report=cleaning_report,
            training_time_seconds=round(time.time() - _train_start, 2),
        )

    baseline_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression()),
        ]
    )

    baseline_pipeline.fit(X_train, y_train)
    baseline_pred = baseline_pipeline.predict(X_test)

    baseline_mse = float(mean_squared_error(y_test, baseline_pred))

    best_reg_pipeline = None
    best_reg_mse = float("inf")
    best_reg_params = None

    for candidate in _regression_rf_candidates():
        candidate_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", RandomForestRegressor(**candidate)),
            ]
        )
        candidate_pipeline.fit(X_train, y_train)
        candidate_pred = candidate_pipeline.predict(X_test)
        candidate_mse = float(mean_squared_error(y_test, candidate_pred))
        if candidate_mse < best_reg_mse:
            best_reg_mse = candidate_mse
            best_reg_pipeline = candidate_pipeline
            best_reg_params = candidate

    final_pipeline = best_reg_pipeline
    final_mse = best_reg_mse

    transformed = final_pipeline.named_steps["preprocessor"].transform(X)
    cluster_plot, cluster_summary = _cluster_payload(transformed)

    trained_preprocessor = final_pipeline.named_steps["preprocessor"]
    rf_model = final_pipeline.named_steps["model"]
    feature_names = _get_feature_names(trained_preprocessor)
    importances = rf_model.feature_importances_

    top_features = sorted(
        [
            {"feature": feature_names[idx], "importance": float(importances[idx])}
            for idx in range(len(feature_names))
        ],
        key=lambda item: item["importance"],
        reverse=True,
    )[:10]

    insight_text = _generate_insight(top_features, "regression")

    metrics = {
        "regression": {
            "linear_regression": {"mse": baseline_mse},
            "random_forest_regressor": {"mse": final_mse},
        },
        "model_comparison": {
            "winner": "Random Forest Regressor" if final_mse <= baseline_mse else "Linear Regression",
            "based_on": "Lower MSE",
            "selected_random_forest_params": best_reg_params,
        },
        "dashboard": {
            "accuracy": None,
            "churn_rate": None,
        },
    }

    bundle = {
        "task_type": "regression",
        "target_column": target_column,
        "pipeline": final_pipeline,
        "expected_features": X.columns.tolist(),
        "column_mapping": column_mapping,
        "feature_importance": top_features,
        "insight_text": insight_text,
        "cleaning_report": cleaning_report,
        "metrics": metrics,
    }
    joblib.dump(bundle, REGRESSOR_MODEL_PATH)

    _last_context.clear()
    _last_context.update(bundle)
    _last_context["cluster_plot"] = cluster_plot
    _last_context["cluster_summary"] = cluster_summary

    return TrainResult(
        task_type="regression",
        target_column=target_column,
        total_records=total_records,
        processed_records=len(clean_df),
        preview=clean_df.head(10).to_dict(orient="records"),
        dataset_rows_sample=clean_df.head(1000).to_dict(orient="records"),
        metrics=metrics,
        cluster_plot=cluster_plot,
        cluster_summary=cluster_summary,
        feature_importance=top_features,
        insight_text=insight_text,
        column_mapping=column_mapping,
        cleaning_report=cleaning_report,
        training_time_seconds=round(time.time() - _train_start, 2),
    )


def _load_latest_bundle(task_type: str):
    if task_type == "classification":
        if not os.path.exists(CLASSIFIER_MODEL_PATH):
            raise ValueError("Classification model not trained yet")
        return joblib.load(CLASSIFIER_MODEL_PATH)

    if task_type == "segmentation":
        if not os.path.exists(SEGMENTATION_MODEL_PATH):
            raise ValueError("Segmentation model not trained yet")
        return joblib.load(SEGMENTATION_MODEL_PATH)

    if not os.path.exists(REGRESSOR_MODEL_PATH):
        raise ValueError("Regression model not trained yet")
    return joblib.load(REGRESSOR_MODEL_PATH)


def predict_record(record: Dict[str, Any], task_type: str = "classification"):
    bundle = _load_latest_bundle(task_type)
    pipeline = bundle["pipeline"]
    expected_features = bundle.get("expected_features", [])

    normalized_expected = {_normalize_name(column): column for column in expected_features}

    aligned_record: Dict[str, Any] = {}
    mapped_fields: Dict[str, str] = {}
    ignored_fields: List[str] = []
    missing_filled: List[str] = []

    for original_key, value in record.items():
        normalized_key = _normalize_name(original_key)
        matched_feature = normalized_expected.get(normalized_key)
        if not matched_feature:
            close = get_close_matches(normalized_key, list(normalized_expected.keys()), n=1, cutoff=0.7)
            if close:
                matched_feature = normalized_expected[close[0]]

        if matched_feature:
            aligned_record[matched_feature] = value
            mapped_fields[original_key] = matched_feature
        else:
            ignored_fields.append(original_key)

    for feature in expected_features:
        if feature not in aligned_record:
            aligned_record[feature] = np.nan
            missing_filled.append(feature)

    prediction_df = pd.DataFrame([aligned_record])
    prediction = pipeline.predict(prediction_df)

    value = prediction[0]
    if isinstance(value, np.generic):
        value = value.item()
    if task_type == "classification":
        probability = None
        if hasattr(pipeline, "predict_proba"):
            probability_values = pipeline.predict_proba(prediction_df)
            probability = float(np.max(probability_values[0]))
        return {
            "prediction": str(value),
            "model_used": "Random Forest",
            "probability_score": probability,
            "insight": bundle.get("insight_text", ""),
            "column_match": {
                "mapped_fields": mapped_fields,
                "missing_filled": missing_filled,
                "ignored_fields": ignored_fields,
            },
        }
    return {
        "prediction": float(value),
        "model_used": "Random Forest Regressor",
        "probability_score": None,
        "insight": bundle.get("insight_text", ""),
        "column_match": {
            "mapped_fields": mapped_fields,
            "missing_filled": missing_filled,
            "ignored_fields": ignored_fields,
        },
    }


def get_feature_importance(task_type: str = "classification"):
    bundle = _load_latest_bundle(task_type)
    return bundle.get("feature_importance", [])


def get_last_metrics(task_type: str = "classification"):
    bundle = _load_latest_bundle(task_type)
    return bundle.get("metrics", {})


def get_insight(task_type: str = "classification"):
    bundle = _load_latest_bundle(task_type)
    return bundle.get("insight_text", "")


def get_last_clusters(task_type: str = "classification"):
    if _last_context.get("task_type") == task_type:
        return {
            "plot": _last_context.get("cluster_plot", {"points": []}),
            "summary": _last_context.get("cluster_summary", []),
        }

    metrics = _load_latest_bundle(task_type)
    return {
        "plot": {"points": []},
        "summary": metrics.get("cluster_summary", []),
    }


def export_metrics_json(task_type: str):
    return json.loads(json.dumps(get_last_metrics(task_type)))
