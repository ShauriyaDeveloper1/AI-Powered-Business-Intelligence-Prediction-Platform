import hashlib
import html
import hmac
import io
import json
import os
import secrets
import base64
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from reportlab.lib import colors  # pyright: ignore[reportMissingImports]
from reportlab.lib.pagesizes import A4  # pyright: ignore[reportMissingImports]
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # pyright: ignore[reportMissingImports]
from reportlab.lib.units import mm  # pyright: ignore[reportMissingImports]
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle  # pyright: ignore[reportMissingImports]

from database import (
    delete_prediction_by_id,
    delete_predictions_by_upload,
    delete_upload_and_related,
    get_analytics_history,
    get_cluster_results,
    get_connection,
    get_latest_upload,
    get_upload_by_id,
    get_upload_history,
    get_prediction_history,
    init_db,
    save_cluster_results,
    save_dataset_rows,
    save_model_metrics,
    save_prediction,
    save_upload,
)
from ml import get_feature_importance, get_insight, get_last_clusters, predict_record, train_models
from schemas import LoginPayload, PredictPayload
from schemas import SignupPayload


app = FastAPI(title="AI Business Intelligence Platform", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TOKEN_TTL_HOURS = int(os.getenv("AI_TOKEN_TTL_HOURS", "12"))
ALLOWED_EXTENSIONS = {".csv"}

VALIDATION_RULES = {
    "churn": {
        "required_columns": ["tenure", "monthlycharges", "totalcharges", "seniorcitizen", "churn"],
        "min_rows": 50,
        "target_column": "churn",
    },
    "sales": {
        "required_columns": ["quantity", "discount", "profit", "sales"],
        "min_rows": 50,
        "target_column": "sales",
    },
    "segmentation": {
        "required_columns": ["tenure", "monthlycharges", "totalcharges"],
        "min_rows": 50,
        "target_column": None,
    },
}


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_column_token(name: Any) -> str:
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())


def _detect_analysis_mode(normalized_columns: List[str]) -> str:
    if "churn" in normalized_columns:
        return "churn"
    if "sales" in normalized_columns:
        return "sales"
    return "segmentation"


def _validate_dataset_by_mode(df: pd.DataFrame, analysis_mode: str, target_column: Optional[str]) -> str:
    mode = analysis_mode if analysis_mode in VALIDATION_RULES else _detect_analysis_mode([_normalize_column_token(col) for col in df.columns])
    rules = VALIDATION_RULES[mode]
    required_columns = rules["required_columns"]
    min_rows = rules["min_rows"]
    expected_target = rules["target_column"]

    normalized_map = {_normalize_column_token(col): str(col) for col in df.columns}
    problems: List[str] = []

    if len(df) < min_rows:
        problems.append(f"Minimum rows required: {min_rows}. Found: {len(df)}.")

    missing = [col for col in required_columns if col not in normalized_map]
    if missing:
        problems.append(f"Missing required columns: {', '.join(missing)}.")

    if expected_target:
        last_column_normalized = _normalize_column_token(df.columns[-1]) if len(df.columns) else ""
        if last_column_normalized != expected_target:
            problems.append(f"Target column must be the last column: {expected_target}.")

        resolved_target = target_column or normalized_map.get(expected_target)
        if resolved_target is None:
            problems.append(f"Target column '{expected_target}' not found.")
        else:
            target_series = df[resolved_target]
            if target_series.isna().any():
                problems.append(f"Target column '{expected_target}' contains missing values.")

            if mode == "churn":
                churn_values = set(pd.to_numeric(target_series, errors="coerce").dropna().astype(int).tolist())
                if churn_values and not churn_values.issubset({0, 1}):
                    problems.append("Churn column must contain only 0 or 1 values.")
                if target_series.isna().any() or pd.to_numeric(target_series, errors="coerce").isna().any():
                    problems.append("Churn column must be numeric and contain only 0 or 1 values.")

            if mode == "sales":
                sales_numeric = pd.to_numeric(target_series, errors="coerce")
                if sales_numeric.isna().any():
                    problems.append("Sales column must be fully numeric.")
                elif (sales_numeric < 0).any():
                    problems.append("Sales values cannot be negative.")

    if mode == "sales":
        for column in required_columns:
            resolved = normalized_map.get(column)
            if not resolved:
                continue
            numeric_series = pd.to_numeric(df[resolved], errors="coerce")
            if numeric_series.isna().any():
                problems.append(f"Column '{column}' must contain only numeric values.")

    if mode == "segmentation":
        for column in required_columns:
            resolved = normalized_map.get(column)
            if not resolved:
                continue
            series = df[resolved]
            if series.isna().any():
                problems.append(f"Column '{column}' contains missing values.")
                continue
            if pd.to_numeric(series, errors="coerce").isna().any():
                problems.append(f"Column '{column}' must contain only numeric values.")

    if mode == "churn":
        for column in required_columns:
            resolved = normalized_map.get(column)
            if not resolved:
                continue
            if df[resolved].isna().any():
                problems.append(f"Column '{column}' contains missing values.")

    if problems:
        required_msg = ", ".join(required_columns)
        problems_text = " ; ".join(dict.fromkeys(problems))
        raise HTTPException(
            status_code=400,
            detail=(
                f"Dataset validation failed for {mode}. "
                f"Required columns: [{required_msg}]. "
                f"Issues: {problems_text}"
            ),
        )

    return mode


def _validate_prediction_dataset(df: pd.DataFrame, task_type: str) -> None:
    mode = "sales" if task_type == "regression" else "segmentation" if task_type == "segmentation" else "churn"

    required_by_mode = {
        "churn": ["tenure", "monthlycharges", "totalcharges", "seniorcitizen"],
        "sales": ["quantity", "discount", "profit"],
        "segmentation": ["tenure", "monthlycharges", "totalcharges"],
    }
    required = required_by_mode[mode]
    normalized_map = {_normalize_column_token(col): str(col) for col in df.columns}
    problems: List[str] = []

    if len(df) == 0:
        problems.append("CSV contains no rows.")

    missing = [col for col in required if col not in normalized_map]
    if missing:
        problems.append(f"Missing required feature columns: {', '.join(missing)}.")

    for column in required:
        resolved = normalized_map.get(column)
        if not resolved:
            continue

        series = df[resolved]
        if series.isna().any():
            problems.append(f"Column '{column}' contains missing values.")
            continue

        if pd.to_numeric(series, errors="coerce").isna().any():
            problems.append(f"Column '{column}' must contain only numeric values.")

    if mode == "sales" and "sales" in normalized_map:
        sales_series = pd.to_numeric(df[normalized_map["sales"]], errors="coerce")
        if sales_series.notna().any() and (sales_series < 0).any():
            problems.append("Sales values cannot be negative.")

    if problems:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Prediction dataset validation failed for {mode}. "
                f"Required feature columns: [{', '.join(required)}]. "
                f"Issues: {' ; '.join(dict.fromkeys(problems))}"
            ),
        )


def _build_improvement_tips(task_type: str, metrics: Dict[str, Any], feature_importance: List[Dict[str, Any]]) -> List[str]:
    tips: List[str] = []

    if task_type == "classification":
        rf_metrics = metrics.get("classification", {}).get("random_forest", {})
        accuracy = _safe_float(rf_metrics.get("accuracy"))
        recall = _safe_float(rf_metrics.get("recall"))
        precision = _safe_float(rf_metrics.get("precision"))

        if accuracy is not None and accuracy < 0.8:
            tips.append("Increase feature quality: review missing values, remove noisy columns, and add business-relevant predictors.")
        if recall is not None and recall < 0.75:
            tips.append("Improve churn-capture recall by balancing classes and prioritizing high-risk segments in training data.")
        if precision is not None and precision < 0.75:
            tips.append("Reduce false positives by tuning decision thresholds and validating top churn indicators.")

    if task_type == "regression":
        rf_metrics = metrics.get("regression", {}).get("random_forest_regressor", {})
        mse = _safe_float(rf_metrics.get("mse"))
        if mse is not None:
            tips.append("Track MSE trend weekly; if MSE increases, retrain with the latest data and review outliers in the target column.")
        tips.append("For sales forecasting, include seasonality drivers (campaign periods, holidays, and product mix) to improve prediction stability.")

    if feature_importance:
        top_names = ", ".join(item.get("feature", "-") for item in feature_importance[:3])
        tips.append(f"Prioritize action plans around top drivers: {top_names}.")

    tips.append("Monitor model performance after each new upload and retrain when business conditions or customer behavior shift.")
    return tips[:6]


def _format_percentage(value: Any) -> str:
    metric = _safe_float(value)
    if metric is None:
        return "-"
    if 0 <= metric <= 1:
        return f"{metric * 100:.2f}%"
    return f"{metric:.2f}"


def _format_number(value: Any) -> str:
    metric = _safe_float(value)
    if metric is None:
        return "-"
    return f"{metric:.4f}"


def _format_rmse_from_mse(value: Any) -> str:
    metric = _safe_float(value)
    if metric is None or metric < 0:
        return "-"
    return f"{metric ** 0.5:.4f}"


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _status_from_rate(rate: float, good_threshold: float, watch_threshold: float) -> str:
    if rate >= good_threshold:
        return "Good"
    if rate >= watch_threshold:
        return "Watch"
    return "Critical"


def _compute_executive_health(task_type: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    if task_type == "classification":
        rf = metrics.get("classification", {}).get("random_forest", {})
        values = [
            _safe_float(rf.get("accuracy")),
            _safe_float(rf.get("precision")),
            _safe_float(rf.get("recall")),
            _safe_float(rf.get("f1")),
        ]
        valid = [v for v in values if v is not None]
        if not valid:
            return {
                "score": "-",
                "status": "Needs Data",
                "notes": "Metrics not available for executive scoring.",
            }

        score = round((sum(valid) / len(valid)) * 100, 1)
        if score >= 85:
            status = "Excellent"
        elif score >= 75:
            status = "Good"
        elif score >= 65:
            status = "Watch"
        else:
            status = "Critical"

        return {
            "score": f"{score:.1f}/100",
            "status": status,
            "notes": "Classification score based on Random Forest accuracy, precision, recall, and F1.",
        }

    lr_mse = _safe_float(metrics.get("regression", {}).get("linear_regression", {}).get("mse"))
    rf_mse = _safe_float(metrics.get("regression", {}).get("random_forest_regressor", {}).get("mse"))

    if lr_mse is None or rf_mse is None or lr_mse <= 0:
        return {
            "score": "-",
            "status": "Needs Data",
            "notes": "Regression MSE comparison not available for executive scoring.",
        }

    improvement = ((lr_mse - rf_mse) / lr_mse) * 100
    if improvement >= 30:
        status = "Excellent"
    elif improvement >= 15:
        status = "Good"
    elif improvement >= 5:
        status = "Watch"
    else:
        status = "Critical"

    return {
        "score": f"{improvement:.1f}% better vs baseline",
        "status": status,
        "notes": "Regression score based on Random Forest MSE improvement over Linear Regression.",
    }


def _build_data_quality_rows(item: Dict[str, Any], metrics: Dict[str, Any]) -> List[List[str]]:
    total = _safe_int(item.get("total_records"))
    processed = _safe_int(item.get("processed_records"))
    dropped = None
    processing_rate = None
    if total is not None and processed is not None and total > 0:
        dropped = max(total - processed, 0)
        processing_rate = (processed / total) * 100

    churn_rate = _safe_float(metrics.get("dashboard", {}).get("churn_rate"))

    rows: List[List[str]] = []
    rows.append(["Total Records", str(total) if total is not None else "-", "Info"])
    rows.append(["Processed Records", str(processed) if processed is not None else "-", "Info"])

    if dropped is not None:
        dropped_pct = (dropped / total) * 100 if total else 0
        status = "Good" if dropped_pct <= 5 else "Watch" if dropped_pct <= 15 else "Critical"
        rows.append(["Dropped Records", f"{dropped} ({dropped_pct:.2f}%)", status])
    else:
        rows.append(["Dropped Records", "-", "Needs Data"])

    if processing_rate is not None:
        status = _status_from_rate(processing_rate, good_threshold=95, watch_threshold=85)
        rows.append(["Processing Rate", f"{processing_rate:.2f}%", status])
    else:
        rows.append(["Processing Rate", "-", "Needs Data"])

    if churn_rate is not None:
        rows.append(["Observed Churn Rate", f"{churn_rate:.2f}%", "Context"])

    return rows


def _build_action_plan(tips: List[str]) -> List[List[str]]:
    priorities = ["P1", "P2", "P3", "P4", "P5", "P6"]
    default_owner = [
        "Data Science Lead",
        "Analytics Team",
        "Business Ops",
        "Product Team",
        "CRM Team",
        "Model Governance",
    ]
    default_timeline = ["1-2 weeks", "2-4 weeks", "This month", "This month", "Quarterly", "Continuous"]

    rows: List[List[str]] = [["Priority", "Recommendation", "Owner", "Timeline"]]
    for idx, tip in enumerate(tips[:6]):
        rows.append([
            priorities[idx],
            tip,
            default_owner[idx],
            default_timeline[idx],
        ])
    return rows


def _normalize_report_mode(report_mode: Optional[str], task_type: str) -> str:
    mode = (report_mode or "").strip().lower()
    if mode in {"segmentation", "sales", "churn"}:
        return mode
    if task_type == "regression":
        return "sales"
    return "churn"


def _build_segmentation_snapshot(clusters: List[Dict[str, Any]], total_records: Optional[int]) -> Dict[str, Any]:
    if not clusters:
        return {
            "segment_count": 0,
            "largest_segment": "-",
            "largest_count": 0,
            "largest_share": "-",
            "coverage": "-",
        }

    sorted_clusters = sorted(clusters, key=lambda item: int(item.get("count", 0)), reverse=True)
    largest = sorted_clusters[0]
    cluster_total = sum(max(int(cluster.get("count", 0)), 0) for cluster in sorted_clusters)
    denominator = total_records if total_records and total_records > 0 else cluster_total

    largest_share = "-"
    coverage = "-"
    if denominator and denominator > 0:
        largest_share = f"{(int(largest.get('count', 0)) / denominator) * 100:.2f}%"
        coverage = f"{(cluster_total / denominator) * 100:.2f}%"

    return {
        "segment_count": len(sorted_clusters),
        "largest_segment": str(largest.get("label", "Cluster")),
        "largest_count": int(largest.get("count", 0)),
        "largest_share": largest_share,
        "coverage": coverage,
    }


def _build_segmentation_action_plan(clusters: List[Dict[str, Any]], feature_importance: List[Dict[str, Any]]) -> List[List[str]]:
    rows: List[List[str]] = [["Priority", "Recommendation", "Owner", "Timeline"]]
    owner_cycle = ["CRM Team", "Marketing Team", "Product Team", "Analytics Team", "Customer Success", "Growth Ops"]
    timeline_cycle = ["1-2 weeks", "2-4 weeks", "This month", "This month", "Quarterly", "Continuous"]

    sorted_clusters = sorted(clusters, key=lambda item: int(item.get("count", 0)), reverse=True)
    top_drivers = ", ".join(str(item.get("feature", "-")).replace("_", " ") for item in feature_importance[:3])
    if top_drivers:
        rows.append([
            "P1",
            f"Design segment-specific campaigns using top drivers: {top_drivers}.",
            owner_cycle[0],
            timeline_cycle[0],
        ])

    for idx, cluster in enumerate(sorted_clusters[:4], start=2):
        label = str(cluster.get("label", f"Cluster {idx - 1}"))
        count = int(cluster.get("count", 0))
        rows.append(
            [
                f"P{idx}",
                f"Create a tailored retention/upsell playbook for '{label}' segment ({count} customers).",
                owner_cycle[(idx - 1) % len(owner_cycle)],
                timeline_cycle[(idx - 1) % len(timeline_cycle)],
            ]
        )

    while len(rows) < 7:
        next_priority = f"P{len(rows)}"
        rows.append(
            [
                next_priority,
                "Track segment movement week-over-week and retrain segmentation when behavior patterns shift.",
                owner_cycle[(len(rows) - 1) % len(owner_cycle)],
                timeline_cycle[(len(rows) - 1) % len(timeline_cycle)],
            ]
        )

    return rows[:7]


def _render_training_report_pdf(
    item: Dict[str, Any],
    metrics: Dict[str, Any],
    feature_importance: List[Dict[str, Any]],
    clusters: List[Dict[str, Any]],
    report_mode: Optional[str] = None,
) -> bytes:
    task_type = item.get("task_type", "classification")
    normalized_report_mode = _normalize_report_mode(report_mode, task_type)
    is_segmentation_report = normalized_report_mode == "segmentation"
    report_title = "Customer Segmentation Intelligence Report" if is_segmentation_report else "AI Business Intelligence Report"
    training_date = item.get("created_at", "-")
    model_used = metrics.get("model_comparison", {}).get("winner", "-")
    tips = _build_improvement_tips(task_type, metrics, feature_importance)
    if is_segmentation_report:
        snapshot = _build_segmentation_snapshot(clusters, _safe_int(item.get("processed_records")))
        executive = {
            "score": f"{snapshot['segment_count']} segments discovered",
            "status": "Good" if snapshot["segment_count"] >= 2 else "Watch",
            "notes": "Segmentation health based on discovered segment diversity and usable cluster coverage.",
        }
    else:
        snapshot = {}
        executive = _compute_executive_health(task_type, metrics)
    data_quality_rows = _build_data_quality_rows(item, metrics)
    action_plan_rows = _build_segmentation_action_plan(clusters, feature_importance) if is_segmentation_report else _build_action_plan(tips)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=report_title,
    )

    styles = getSampleStyleSheet()
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        textColor=colors.HexColor("#0b3a53"),
        spaceAfter=6,
        spaceBefore=10,
    )
    normal_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
    )

    small_style = ParagraphStyle(
        "Small",
        parent=normal_style,
        fontSize=8,
        textColor=colors.HexColor("#6b7280"),
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=normal_style,
        fontSize=8.5,
        leading=11,
    )

    table_cell_bold_style = ParagraphStyle(
        "TableCellBold",
        parent=table_cell_style,
        fontName="Helvetica-Bold",
    )

    def _safe_text(value: Any) -> str:
        return html.escape(str(value))

    def _status_color(status: str):
        palette = {
            "Excellent": colors.HexColor("#14532d"),
            "Good": colors.HexColor("#166534"),
            "Watch": colors.HexColor("#9a3412"),
            "Critical": colors.HexColor("#991b1b"),
            "Needs Data": colors.HexColor("#6b7280"),
        }
        return palette.get(status, colors.HexColor("#374151"))

    def _decorate_status_column(table: Table, rows_len: int, col_index: int = 2):
        for row_idx in range(1, rows_len):
            status = table._cellvalues[row_idx][col_index]
            table.setStyle(
                TableStyle(
                    [
                        ("TEXTCOLOR", (col_index, row_idx), (col_index, row_idx), _status_color(str(status))),
                        ("FONTNAME", (col_index, row_idx), (col_index, row_idx), "Helvetica-Bold"),
                    ]
                )
            )

    def _draw_page_frame(canvas, pdf_doc):
        canvas.saveState()
        page_w, page_h = A4
        canvas.setStrokeColor(colors.HexColor("#d1d5db"))
        canvas.setLineWidth(0.6)
        canvas.rect(12 * mm, 12 * mm, page_w - 24 * mm, page_h - 24 * mm, stroke=1, fill=0)

        canvas.setFont("Helvetica-Bold", 9)
        canvas.setFillColor(colors.HexColor("#0b3a53"))
        canvas.drawString(18 * mm, page_h - 14 * mm, report_title)

        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#6b7280"))
        canvas.drawRightString(page_w - 18 * mm, page_h - 14 * mm, f"Generated: {generated_at}")
        canvas.drawString(18 * mm, 10 * mm, "Confidential - Internal decision support document")
        canvas.drawRightString(page_w - 18 * mm, 10 * mm, f"Page {canvas.getPageNumber()}")
        canvas.restoreState()

    story: List[Any] = []
    story.append(Paragraph(report_title, styles["Title"]))
    story.append(Paragraph(f"Generated: {_safe_text(generated_at)}", small_style))
    story.append(Spacer(1, 8))

    story.append(Paragraph("0. Executive Summary", heading_style))
    exec_rows = [
        ["Overall Health Score", str(executive["score"])],
        ["Health Status", str(executive["status"])],
        ["Method", str(executive["notes"])],
    ]
    exec_table = Table(exec_rows, colWidths=[48 * mm, 120 * mm])
    exec_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef2ff")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#1f2937")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    exec_table.setStyle(
        TableStyle(
            [
                ("TEXTCOLOR", (1, 1), (1, 1), _status_color(str(executive["status"]))),
                ("FONTNAME", (1, 1), (1, 1), "Helvetica-Bold"),
            ]
        )
    )
    story.append(exec_table)

    summary_rows = [
        ["Dataset Name", _safe_text(item.get("filename", "-"))],
        ["Analysis Type", "Customer Segmentation" if is_segmentation_report else _safe_text(task_type)],
        ["Target Column", _safe_text(item.get("target_column") or "-")],
        ["Training Date", _safe_text(training_date)],
        ["Total Records", _safe_text(item.get("total_records", "-"))],
        ["Processed Records", _safe_text(item.get("processed_records", "-"))],
        ["Winning Model", _safe_text(model_used)],
    ]

    story.append(Paragraph("1. Training Summary", heading_style))
    summary_table = Table(summary_rows, colWidths=[48 * mm, 120 * mm])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f4f6f8")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#1f2937")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(summary_table)

    if is_segmentation_report:
        story.append(Paragraph("2. Segment Snapshot", heading_style))
        snapshot_rows = [
            ["Metric", "Value", "Status"],
            ["Segments Discovered", str(snapshot.get("segment_count", 0)), "Good" if snapshot.get("segment_count", 0) >= 2 else "Watch"],
            ["Largest Segment", str(snapshot.get("largest_segment", "-")), "Context"],
            ["Largest Segment Size", str(snapshot.get("largest_count", "-")), "Info"],
            ["Largest Segment Share", str(snapshot.get("largest_share", "-")), "Context"],
            ["Cluster Coverage", str(snapshot.get("coverage", "-")), "Good"],
        ]
        snapshot_table = Table(snapshot_rows, colWidths=[72 * mm, 72 * mm, 24 * mm])
        snapshot_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b3a53")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
                    ("ALIGN", (2, 1), (2, -1), "CENTER"),
                ]
            )
        )
        _decorate_status_column(snapshot_table, len(snapshot_rows), col_index=2)
        story.append(snapshot_table)
    else:
        story.append(Paragraph("2. Performance Metrics", heading_style))
        if task_type == "classification":
            lr = metrics.get("classification", {}).get("logistic_regression", {})
            rf = metrics.get("classification", {}).get("random_forest", {})
            metrics_rows = [
                ["Model", "Accuracy", "Precision", "Recall", "F1 Score"],
                [
                    "Logistic Regression",
                    _format_percentage(lr.get("accuracy")),
                    _format_percentage(lr.get("precision")),
                    _format_percentage(lr.get("recall")),
                    _format_percentage(lr.get("f1")),
                ],
                [
                    "Random Forest",
                    _format_percentage(rf.get("accuracy")),
                    _format_percentage(rf.get("precision")),
                    _format_percentage(rf.get("recall")),
                    _format_percentage(rf.get("f1")),
                ],
            ]
        else:
            lr = metrics.get("regression", {}).get("linear_regression", {})
            rf = metrics.get("regression", {}).get("random_forest_regressor", {})
            metrics_rows = [
                ["Model", "MSE", "RMSE (approx)", "Status"],
                [
                    "Linear Regression",
                    _format_number(lr.get("mse")),
                    _format_rmse_from_mse(lr.get("mse")),
                    "Baseline",
                ],
                [
                    "Random Forest Regressor",
                    _format_number(rf.get("mse")),
                    _format_rmse_from_mse(rf.get("mse")),
                    "Preferred" if model_used == "random_forest_regressor" else "Candidate",
                ],
            ]

        perf_table = Table(metrics_rows)
        perf_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b3a53")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
                    ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ]
            )
        )
        story.append(perf_table)

    story.append(Paragraph("2.1 Data Quality & Processing", heading_style))
    quality_table_rows = [["Indicator", "Value", "Status"], *data_quality_rows]
    quality_table = Table(quality_table_rows, colWidths=[70 * mm, 75 * mm, 23 * mm])
    quality_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f4f6f8")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
                ("ALIGN", (2, 1), (2, -1), "CENTER"),
            ]
        )
    )
    _decorate_status_column(quality_table, len(quality_table_rows), col_index=2)
    story.append(quality_table)

    story.append(Paragraph("3. Key Segment Drivers" if is_segmentation_report else "3. Top Feature Drivers", heading_style))
    if feature_importance:
        feature_rows = [["Rank", "Feature", "Importance"]]
        for idx, feat in enumerate(feature_importance[:10], start=1):
            feature_rows.append([
                str(idx),
                _safe_text(feat.get("feature", "-")),
                _format_number(feat.get("importance")),
            ])
        feature_table = Table(feature_rows, colWidths=[20 * mm, 120 * mm, 28 * mm])
        feature_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f4f6f8")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
                ]
            )
        )
        story.append(feature_table)
    else:
        story.append(Paragraph("No feature importance available for this run.", normal_style))

    story.append(Paragraph("4. Segment Distribution" if is_segmentation_report else "4. Cluster Distribution", heading_style))
    if clusters:
        cluster_rows = [["Cluster", "Records", "Share"]] if is_segmentation_report else [["Cluster", "Records"]]
        processed_records = _safe_int(item.get("processed_records")) or 0
        cluster_total = sum(max(int(cluster.get("count", 0)), 0) for cluster in clusters)
        denominator = processed_records if processed_records > 0 else cluster_total
        for cluster in clusters:
            count = int(cluster.get("count", 0))
            share = f"{(count / denominator) * 100:.2f}%" if denominator > 0 else "-"
            if is_segmentation_report:
                cluster_rows.append([
                    Paragraph(_safe_text(cluster.get("label", "Cluster")), table_cell_style),
                    Paragraph(str(count), table_cell_style),
                    Paragraph(share, table_cell_style),
                ])
            else:
                cluster_rows.append([
                    Paragraph(_safe_text(cluster.get("label", "Cluster")), table_cell_style),
                    Paragraph(str(count), table_cell_style),
                ])
        cluster_table = Table(cluster_rows, colWidths=[100 * mm, 30 * mm, 38 * mm]) if is_segmentation_report else Table(cluster_rows, colWidths=[118 * mm, 50 * mm])
        cluster_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f4f6f8")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
                    ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 7),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(cluster_table)
    else:
        story.append(Paragraph("No cluster summary available for this run.", normal_style))

    story.append(Paragraph("5. Segmentation Insight" if is_segmentation_report else "5. Business Insight", heading_style))
    story.append(Paragraph(_safe_text(item.get("insight_text") or "No insight generated."), normal_style))

    story.append(Paragraph("6. Segment Action Plan" if is_segmentation_report else "6. Priority Action Plan", heading_style))
    action_rows: List[List[Any]] = [
        [
            action_plan_rows[0][0],
            action_plan_rows[0][1],
            action_plan_rows[0][2],
            action_plan_rows[0][3],
        ]
    ]
    for row in action_plan_rows[1:]:
        action_rows.append(
            [
                Paragraph(_safe_text(row[0]), table_cell_bold_style),
                Paragraph(_safe_text(row[1]), table_cell_style),
                Paragraph(_safe_text(row[2]), table_cell_style),
                Paragraph(_safe_text(row[3]), table_cell_style),
            ]
        )

    action_table = Table(action_rows, colWidths=[14 * mm, 86 * mm, 46 * mm, 22 * mm])
    action_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b3a53")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (0, 1), (0, -1), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(action_table)

    story.append(Spacer(1, 10))
    if is_segmentation_report:
        note_text = "Report Note: Segment actions are generated from cluster distribution and top segment drivers from the latest run."
    else:
        note_text = "Report Note: Scores and recommendations are generated from the latest uploaded dataset and model outputs."
    story.append(Paragraph(note_text, small_style))

    doc.build(story, onFirstPage=_draw_page_frame, onLaterPages=_draw_page_frame)
    return buffer.getvalue()


def hash_password(raw_password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", raw_password.encode("utf-8"), salt, 120000)
    return f"pbkdf2_sha256${base64.b64encode(salt).decode()}${base64.b64encode(digest).decode()}"


def verify_password(raw_password: str, stored_hash: str) -> bool:
    if stored_hash.startswith("pbkdf2_sha256$"):
        _, b64_salt, b64_digest = stored_hash.split("$", 2)
        salt = base64.b64decode(b64_salt.encode())
        expected = base64.b64decode(b64_digest.encode())
        current = hashlib.pbkdf2_hmac("sha256", raw_password.encode("utf-8"), salt, 120000)
        return hmac.compare_digest(current, expected)

    legacy = hashlib.sha256(raw_password.encode("utf-8")).hexdigest()
    return hmac.compare_digest(legacy, stored_hash)


def issue_token(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    expires_at = (datetime.now(timezone.utc) + timedelta(hours=TOKEN_TTL_HOURS)).isoformat()

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO auth_tokens (token, user_id, expires_at, created_at) VALUES (%s, %s, %s, %s)",
        (
            token,
            user_id,
            expires_at,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    cursor.close()
    conn.close()
    return token


def validate_token(authorization: Optional[str] = Header(default=None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid auth token")

    token = authorization.replace("Bearer ", "", 1).strip()
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        """
        SELECT t.token, t.user_id, t.expires_at, u.username, u.role
        FROM auth_tokens t
        JOIN users u ON u.id = t.user_id
        WHERE t.token = %s
        """,
        (token,),
    )
    row = cursor.fetchone()
    cursor.close()
    conn.close()

    if not row:
        raise HTTPException(status_code=401, detail="Invalid token")

    expires_at = datetime.fromisoformat(row["expires_at"])
    if datetime.now(timezone.utc) > expires_at:
        raise HTTPException(status_code=401, detail="Token expired")

    return {"user_id": row["user_id"], "username": row["username"], "role": row["role"]}


@app.on_event("startup")
def startup_event():
    init_db()


@app.get("/")
def health_check():
    return {"message": "AI Business Platform Running", "secure": True}


@app.post("/auth/login")
def login(payload: LoginPayload):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, username, password_hash, role FROM users WHERE username = %s", (payload.username,))
    user = cursor.fetchone()
    cursor.close()
    conn.close()

    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not user["password_hash"].startswith("pbkdf2_sha256$"):
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE users SET password_hash = %s WHERE id = %s",
            (hash_password(payload.password), user["id"]),
        )
        conn.commit()
        cursor.close()
        conn.close()

    token = issue_token(user["id"])

    return {
        "message": "Login successful",
        "token": token,
        "user": {"username": user["username"], "role": user["role"]},
    }


@app.post("/auth/signup")
def signup(payload: SignupPayload):
    username = payload.username.strip()
    if not username:
        raise HTTPException(status_code=400, detail="Username is required")
    if len(username) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters")
    if payload.password != payload.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match")
    if len(payload.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
    exists = cursor.fetchone()
    if exists:
        cursor.close()
        conn.close()
        raise HTTPException(status_code=409, detail="User ID already exists")

    cursor.close()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO users (username, password_hash, role, created_at) VALUES (%s, %s, %s, %s)",
        (
            username,
            hash_password(payload.password),
            "user",
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    cursor.close()
    conn.close()
    return {
        "message": "Signup successful. Please login.",
        "user": {"username": username, "role": "user"},
    }


@app.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    task_type: str = "classification",
    analysis_mode: Optional[str] = None,
    target_column: Optional[str] = None,
    user_ctx: Dict[str, Any] = Depends(validate_token),
):
    extension = os.path.splitext(file.filename or "")[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    if not file.filename:
        raise HTTPException(status_code=400, detail="File name is missing")

    os.makedirs(os.path.join(os.path.dirname(__file__), "data", "uploads"), exist_ok=True)
    saved_path = os.path.join(
        os.path.dirname(__file__),
        "data",
        "uploads",
        f"{int(datetime.now().timestamp())}_{file.filename}",
    )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    with open(saved_path, "wb") as buffer:
        buffer.write(content)

    try:
        validation_df = pd.read_csv(io.BytesIO(content), keep_default_na=True, na_values=["", " ", "na", "n/a", "null", "none", "nan", "-", "?"])
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(exc)}") from exc

    mode_hint = (analysis_mode or "").strip().lower()
    if mode_hint not in {"churn", "sales", "segmentation"}:
        mode_hint = "sales" if task_type == "regression" else "churn"
    resolved_mode = _validate_dataset_by_mode(validation_df, mode_hint, target_column)

    if resolved_mode == "segmentation":
        task_type = "segmentation"
    elif resolved_mode == "sales":
        task_type = "regression"
    else:
        task_type = "classification"

    if resolved_mode == "churn" and not target_column:
        target_column = "churn"
    if resolved_mode == "sales" and not target_column:
        target_column = "sales"

    try:
        result = train_models(saved_path, task_type=task_type, target_column=target_column)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    upload_id = save_upload(
        user_id=user_ctx["user_id"],
        filename=file.filename,
        storage_path=saved_path,
        task_type=result.task_type,
        target_column=result.target_column,
        total_records=result.total_records,
        processed_records=result.processed_records,
        metrics=result.metrics,
        feature_importance=result.feature_importance,
        insight_text=result.insight_text,
        preview=result.preview,
    )
    save_model_metrics(upload_id, result.task_type, result.metrics)
    save_cluster_results(upload_id, result.task_type, result.cluster_summary)

    save_dataset_rows(upload_id, result.dataset_rows_sample)

    return {
        "message": "Dataset uploaded and pipeline executed successfully",
        "upload_id": upload_id,
        "analysis_mode": resolved_mode,
        "task_type": result.task_type,
        "target_column": result.target_column,
        "total_records": result.total_records,
        "processed_records": result.processed_records,
        "training_time_seconds": result.training_time_seconds,
        "preview": result.preview,
        "metrics": result.metrics,
        "clusters": {
            "plot": result.cluster_plot,
            "summary": result.cluster_summary,
        },
        "feature_importance": result.feature_importance,
        "insight": result.insight_text,
        "column_mapping": result.column_mapping,
        "cleaning_report": result.cleaning_report,
    }


@app.post("/predict")
def predict(payload: PredictPayload, user_ctx: Dict[str, Any] = Depends(validate_token)):
    try:
        pred_result = predict_record(payload.record, task_type=payload.task_type)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    prediction_id = save_prediction(
        user_id=user_ctx["user_id"],
        upload_id=payload.upload_id,
        task_type=payload.task_type,
        input_data=payload.record,
        output_value=pred_result["prediction"],
        model_used=pred_result["model_used"],
        probability_score=pred_result["probability_score"],
        insight_text=pred_result["insight"],
    )

    return {
        "prediction_id": prediction_id,
        "prediction": pred_result["prediction"],
        "model_used": pred_result["model_used"],
        "probability_score": pred_result["probability_score"],
        "insight": pred_result["insight"],
        "column_match": pred_result.get("column_match", {}),
    }


@app.post("/predict-upload")
async def predict_upload(
    file: UploadFile = File(...),
    task_type: str = "classification",
    upload_id: Optional[int] = None,
    user_ctx: Dict[str, Any] = Depends(validate_token),
):
    extension = os.path.splitext(file.filename or "")[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    if not file.filename:
        raise HTTPException(status_code=400, detail="File name is missing")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(exc)}") from exc

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV contains no rows")

    _validate_prediction_dataset(df, task_type)

    output_rows: List[Dict[str, Any]] = []
    model_used = ""
    insight = ""
    max_preview = 200

    try:
        for index, row in df.iterrows():
            record = row.where(pd.notna(row), None).to_dict()
            pred_result = predict_record(record, task_type=task_type)

            model_used = pred_result.get("model_used", model_used)
            insight = pred_result.get("insight", insight)

            save_prediction(
                user_id=user_ctx["user_id"],
                upload_id=upload_id,
                task_type=task_type,
                input_data=record,
                output_value=pred_result["prediction"],
                model_used=pred_result["model_used"],
                probability_score=pred_result["probability_score"],
                insight_text=pred_result["insight"],
            )

            if len(output_rows) < max_preview:
                output_rows.append(
                    {
                        "row": int(index + 1),
                        "prediction": pred_result["prediction"],
                        "probability": pred_result["probability_score"],
                        "model": pred_result["model_used"],
                    }
                )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "message": "Batch prediction completed",
        "filename": file.filename,
        "task_type": task_type,
        "predicted_rows": int(len(df)),
        "preview_rows": int(len(output_rows)),
        "model_used": model_used,
        "insight": insight,
        "preview": output_rows,
    }


@app.get("/feature-importance")
def feature_importance(task_type: str = "classification", _: Dict[str, Any] = Depends(validate_token)):
    try:
        importance = get_feature_importance(task_type)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"task_type": task_type, "importance": importance}


@app.get("/clusters")
def cluster_data(task_type: str = "classification", _: Dict[str, Any] = Depends(validate_token)):
    try:
        clusters = get_last_clusters(task_type)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"task_type": task_type, **clusters}


@app.get("/insights")
def insights(task_type: str = "classification", _: Dict[str, Any] = Depends(validate_token)):
    try:
        importance = get_feature_importance(task_type)
        insight = get_insight(task_type)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "task_type": task_type,
        "top_features": importance[:5],
        "insight": insight,
    }


@app.get("/dashboard")
def dashboard(user_ctx: Dict[str, Any] = Depends(validate_token)):
    latest = get_latest_upload(user_ctx["user_id"])
    if not latest:
        return {
            "total_records": 0,
            "model_accuracy": None,
            "churn_rate": None,
            "cluster_distribution": [],
            "feature_importance": [],
            "model_comparison": {},
        }

    metrics = json.loads(latest["metrics_json"])
    clusters = get_cluster_results(latest["id"])
    if not clusters:
        clusters = get_last_clusters(latest["task_type"]).get("summary", [])

    model_accuracy = None
    if latest["task_type"] == "classification":
        model_accuracy = metrics.get("classification", {}).get("random_forest", {}).get("accuracy")

    return {
        "upload_id": latest["id"],
        "task_type": latest["task_type"],
        "total_records": latest["total_records"],
        "processed_records": latest["processed_records"],
        "model_accuracy": model_accuracy,
        "churn_rate": metrics.get("dashboard", {}).get("churn_rate"),
        "cluster_distribution": clusters,
        "feature_importance": json.loads(latest.get("feature_importance_json") or "[]"),
        "insight": latest.get("insight_text") or "",
        "model_comparison": metrics.get("model_comparison", {}),
        "metrics": metrics,
    }


@app.get("/history")
def history(
    limit: int = 30,
    upload_id: Optional[int] = None,
    user_ctx: Dict[str, Any] = Depends(validate_token),
):
    rows = get_prediction_history(user_ctx["user_id"], limit=limit, upload_id=upload_id)
    for row in rows:
        row["input"] = json.loads(row["input_json"])
        row.pop("input_json", None)
    return {"items": rows}


@app.delete("/history/predictions/{prediction_id}")
def delete_prediction_item(prediction_id: int, user_ctx: Dict[str, Any] = Depends(validate_token)):
    deleted = delete_prediction_by_id(prediction_id, user_ctx["user_id"])
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return {"message": "Prediction deleted", "prediction_id": prediction_id, "deleted": deleted}


@app.delete("/history/upload/{upload_id}")
def delete_prediction_group(upload_id: int, user_ctx: Dict[str, Any] = Depends(validate_token)):
    deleted = delete_predictions_by_upload(upload_id, user_ctx["user_id"])
    if deleted == 0:
        raise HTTPException(status_code=404, detail="No predictions found for this dataset")
    return {"message": "Prediction history deleted for dataset", "upload_id": upload_id, "deleted": deleted}


@app.get("/latest-upload")
def latest_upload(user_ctx: Dict[str, Any] = Depends(validate_token)):
    latest = get_latest_upload(user_ctx["user_id"])
    if not latest:
        return {"item": None}

    latest["metrics"] = json.loads(latest["metrics_json"])
    latest["feature_importance"] = json.loads(latest.get("feature_importance_json") or "[]")
    latest["preview"] = json.loads(latest["preview_json"])
    latest.pop("metrics_json", None)
    latest.pop("feature_importance_json", None)
    latest.pop("preview_json", None)
    return {"item": latest}


@app.get("/uploads")
def upload_history(limit: int = 20, user_ctx: Dict[str, Any] = Depends(validate_token)):
    rows = get_upload_history(user_ctx["user_id"], limit=limit)
    return {"items": rows}


@app.get("/analytics-history")
def analytics_history(limit: int = 20, user_ctx: Dict[str, Any] = Depends(validate_token)):
    items = get_analytics_history(user_ctx["user_id"], limit=limit)
    return {"items": items}


@app.get("/uploads/{upload_id}")
def upload_analysis(upload_id: int, user_ctx: Dict[str, Any] = Depends(validate_token)):
    item = get_upload_by_id(upload_id, user_ctx["user_id"])
    if not item:
        raise HTTPException(status_code=404, detail="Upload not found")

    metrics = json.loads(item["metrics_json"])
    feature_importance = json.loads(item.get("feature_importance_json") or "[]")
    preview = json.loads(item["preview_json"])
    clusters = get_cluster_results(upload_id)

    if not clusters:
        clusters = get_last_clusters(item["task_type"]).get("summary", [])

    item.pop("metrics_json", None)
    item.pop("feature_importance_json", None)
    item.pop("preview_json", None)

    return {
        "item": {
            **item,
            "metrics": metrics,
            "feature_importance": feature_importance,
            "insight": item.get("insight_text") or "",
            "preview": preview,
            "clusters": {
                "summary": clusters,
            },
        }
    }


@app.get("/uploads/{upload_id}/report")
def download_upload_report(
    upload_id: int,
    report_mode: Optional[str] = None,
    user_ctx: Dict[str, Any] = Depends(validate_token),
):
    item = get_upload_by_id(upload_id, user_ctx["user_id"])
    if not item:
        raise HTTPException(status_code=404, detail="Upload not found")

    metrics = json.loads(item.get("metrics_json") or "{}")
    feature_importance = json.loads(item.get("feature_importance_json") or "[]")
    clusters = get_cluster_results(upload_id)

    report_bytes = _render_training_report_pdf(item, metrics, feature_importance, clusters, report_mode=report_mode)

    dataset_name = str(item.get("filename") or "dataset")
    safe_name = "".join(ch for ch in os.path.splitext(dataset_name)[0] if ch.isalnum() or ch in ("-", "_"))
    if not safe_name:
        safe_name = f"dataset_{upload_id}"

    headers = {
        "Content-Disposition": f'attachment; filename="ai_report_{upload_id}_{safe_name}.pdf"'
    }
    return Response(content=report_bytes, media_type="application/pdf", headers=headers)


@app.delete("/uploads/{upload_id}")
def delete_upload(upload_id: int, user_ctx: Dict[str, Any] = Depends(validate_token)):
    deleted = delete_upload_and_related(upload_id, user_ctx["user_id"])
    if not deleted or deleted.get("deleted_uploads", 0) == 0:
        raise HTTPException(status_code=404, detail="Upload not found")

    file_deleted = False
    storage_path = deleted.get("storage_path")
    if storage_path:
        try:
            if os.path.isfile(storage_path):
                os.remove(storage_path)
                file_deleted = True
        except Exception:
            file_deleted = False
    else:
        try:
            uploads_dir = os.path.join(os.path.dirname(__file__), "data", "uploads")
            if os.path.isdir(uploads_dir):
                suffix = f"_{deleted['filename']}"
                for candidate in os.listdir(uploads_dir):
                    if candidate.endswith(suffix):
                        candidate_path = os.path.join(uploads_dir, candidate)
                        if os.path.isfile(candidate_path):
                            os.remove(candidate_path)
                            file_deleted = True
                            break
        except Exception:
            file_deleted = False

    return {
        "message": "Dataset and related analytics deleted",
        "upload_id": deleted["id"],
        "filename": deleted["filename"],
        "deleted_predictions": deleted.get("deleted_predictions", 0),
        "file_deleted": file_deleted,
    }
