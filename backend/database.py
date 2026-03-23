import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import mysql.connector

DB_HOST = os.getenv("AI_DB_HOST", "localhost")
DB_PORT = int(os.getenv("AI_DB_PORT", "3306"))
DB_USER = os.getenv("AI_DB_USER", "root")
DB_PASSWORD = os.getenv("AI_DB_PASSWORD", "24#2006")
DB_NAME = os.getenv("AI_DB_NAME", "ai_platform")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _base_config():
    return {
        "host": DB_HOST,
        "port": DB_PORT,
        "user": DB_USER,
        "password": DB_PASSWORD,
    }


def get_connection():
    config = _base_config()
    config["database"] = DB_NAME
    return mysql.connector.connect(**config)


def _dict_cursor(conn):
    return conn.cursor(dictionary=True)


def _column_exists(cursor, table_name: str, column_name: str) -> bool:
    cursor.execute(
        """
        SELECT 1
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s AND COLUMN_NAME = %s
        LIMIT 1
        """,
        (DB_NAME, table_name, column_name),
    )
    return cursor.fetchone() is not None


def _ensure_column(cursor, table_name: str, column_name: str, column_definition: str):
    if _column_exists(cursor, table_name, column_name):
        return
    cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}")


def init_db():
    admin_conn = mysql.connector.connect(**_base_config())
    admin_cursor = admin_conn.cursor()
    admin_cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
    admin_conn.commit()
    admin_cursor.close()
    admin_conn.close()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INT PRIMARY KEY AUTO_INCREMENT,
            username VARCHAR(100) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            role VARCHAR(50) NOT NULL DEFAULT 'admin',
            created_at TEXT NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS auth_tokens (
            token VARCHAR(255) PRIMARY KEY,
            user_id INT NOT NULL,
            expires_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
                ON DELETE CASCADE
                ON UPDATE CASCADE
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS uploads (
            id INT PRIMARY KEY AUTO_INCREMENT,
            user_id INT NULL,
            filename VARCHAR(255) NOT NULL,
            task_type VARCHAR(40) NOT NULL,
            target_column VARCHAR(255) NOT NULL,
            total_records INT NOT NULL,
            processed_records INT NOT NULL,
            metrics_json LONGTEXT NOT NULL,
            feature_importance_json LONGTEXT NULL,
            insight_text TEXT NULL,
            preview_json LONGTEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
                ON DELETE SET NULL
                ON UPDATE CASCADE
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS dataset_rows (
            id INT PRIMARY KEY AUTO_INCREMENT,
            upload_id INT NOT NULL,
            row_index INT NOT NULL,
            row_json LONGTEXT NOT NULL,
            FOREIGN KEY (upload_id) REFERENCES uploads(id)
                ON DELETE CASCADE
                ON UPDATE CASCADE
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INT PRIMARY KEY AUTO_INCREMENT,
            user_id INT NULL,
            upload_id INT NULL,
            task_type VARCHAR(40) NOT NULL,
            input_json LONGTEXT NOT NULL,
            output_value TEXT NOT NULL,
            model_used VARCHAR(100) NOT NULL,
            probability_score FLOAT NULL,
            insight_text TEXT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
                ON DELETE SET NULL
                ON UPDATE CASCADE,
            FOREIGN KEY (upload_id) REFERENCES uploads(id)
                ON DELETE SET NULL
                ON UPDATE CASCADE
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS model_metrics (
            id INT PRIMARY KEY AUTO_INCREMENT,
            upload_id INT NOT NULL,
            task_type VARCHAR(40) NOT NULL,
            model_name VARCHAR(100) NOT NULL,
            metric_key VARCHAR(100) NOT NULL,
            metric_value DOUBLE NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (upload_id) REFERENCES uploads(id)
                ON DELETE CASCADE
                ON UPDATE CASCADE
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS cluster_results (
            id INT PRIMARY KEY AUTO_INCREMENT,
            upload_id INT NOT NULL,
            task_type VARCHAR(40) NOT NULL,
            cluster_id INT NOT NULL,
            cluster_label VARCHAR(150) NOT NULL,
            cluster_count INT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (upload_id) REFERENCES uploads(id)
                ON DELETE CASCADE
                ON UPDATE CASCADE
        )
        """
    )

    _ensure_column(cursor, "users", "role", "VARCHAR(50) NULL DEFAULT 'admin'")
    _ensure_column(cursor, "users", "created_at", "TEXT NULL")

    _ensure_column(cursor, "auth_tokens", "expires_at", "TEXT NULL")
    _ensure_column(cursor, "auth_tokens", "created_at", "TEXT NULL")

    _ensure_column(cursor, "uploads", "task_type", "VARCHAR(40) NULL")
    _ensure_column(cursor, "uploads", "user_id", "INT NULL")
    _ensure_column(cursor, "uploads", "target_column", "VARCHAR(255) NULL")
    _ensure_column(cursor, "uploads", "total_records", "INT NULL")
    _ensure_column(cursor, "uploads", "processed_records", "INT NULL")
    _ensure_column(cursor, "uploads", "metrics_json", "LONGTEXT NULL")
    _ensure_column(cursor, "uploads", "feature_importance_json", "LONGTEXT NULL")
    _ensure_column(cursor, "uploads", "insight_text", "TEXT NULL")
    _ensure_column(cursor, "uploads", "preview_json", "LONGTEXT NULL")
    _ensure_column(cursor, "uploads", "created_at", "TEXT NULL")
    _ensure_column(cursor, "uploads", "storage_path", "TEXT NULL")

    _ensure_column(cursor, "dataset_rows", "upload_id", "INT NULL")
    _ensure_column(cursor, "dataset_rows", "row_index", "INT NULL")
    _ensure_column(cursor, "dataset_rows", "row_json", "LONGTEXT NULL")

    _ensure_column(cursor, "predictions", "upload_id", "INT NULL")
    _ensure_column(cursor, "predictions", "user_id", "INT NULL")
    _ensure_column(cursor, "predictions", "task_type", "VARCHAR(40) NULL")
    _ensure_column(cursor, "predictions", "input_json", "LONGTEXT NULL")
    _ensure_column(cursor, "predictions", "output_value", "TEXT NULL")
    _ensure_column(cursor, "predictions", "model_used", "VARCHAR(100) NULL")
    _ensure_column(cursor, "predictions", "probability_score", "FLOAT NULL")
    _ensure_column(cursor, "predictions", "insight_text", "TEXT NULL")
    _ensure_column(cursor, "predictions", "created_at", "TEXT NULL")

    _ensure_column(cursor, "model_metrics", "upload_id", "INT NULL")
    _ensure_column(cursor, "model_metrics", "task_type", "VARCHAR(40) NULL")
    _ensure_column(cursor, "model_metrics", "model_name", "VARCHAR(100) NULL")
    _ensure_column(cursor, "model_metrics", "metric_key", "VARCHAR(100) NULL")
    _ensure_column(cursor, "model_metrics", "metric_value", "DOUBLE NULL")
    _ensure_column(cursor, "model_metrics", "created_at", "TEXT NULL")

    _ensure_column(cursor, "cluster_results", "upload_id", "INT NULL")
    _ensure_column(cursor, "cluster_results", "task_type", "VARCHAR(40) NULL")
    _ensure_column(cursor, "cluster_results", "cluster_id", "INT NULL")
    _ensure_column(cursor, "cluster_results", "cluster_label", "VARCHAR(150) NULL")
    _ensure_column(cursor, "cluster_results", "cluster_count", "INT NULL")
    _ensure_column(cursor, "cluster_results", "created_at", "TEXT NULL")

    conn.commit()
    cursor.close()
    conn.close()


def save_upload(
    user_id,
    filename,
    storage_path,
    task_type,
    target_column,
    total_records,
    processed_records,
    metrics,
    feature_importance,
    insight_text,
    preview,
):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO uploads (
            user_id, filename, storage_path, task_type, target_column, total_records,
            processed_records, metrics_json, feature_importance_json, insight_text, preview_json, created_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            user_id,
            filename,
            storage_path,
            task_type,
            target_column,
            total_records,
            processed_records,
            json.dumps(metrics),
            json.dumps(feature_importance),
            insight_text,
            json.dumps(preview),
            _utc_now(),
        ),
    )
    upload_id = cursor.lastrowid
    conn.commit()
    cursor.close()
    conn.close()
    return upload_id


def save_dataset_rows(upload_id, rows):
    conn = get_connection()
    cursor = conn.cursor()
    payload = [(upload_id, idx, json.dumps(row)) for idx, row in enumerate(rows)]
    cursor.executemany(
        "INSERT INTO dataset_rows (upload_id, row_index, row_json) VALUES (%s, %s, %s)",
        payload,
    )
    conn.commit()
    cursor.close()
    conn.close()


def _flatten_metrics(metrics: Dict[str, Any]) -> List[tuple]:
    flattened = []

    def walk(node: Any, path: List[str]):
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, path + [key])
            return
        if isinstance(node, (int, float)):
            if len(path) >= 2:
                model_name = path[-2]
                metric_key = path[-1]
            elif len(path) == 1:
                model_name = "global"
                metric_key = path[0]
            else:
                model_name = "global"
                metric_key = "value"
            flattened.append((model_name, metric_key, float(node)))

    walk(metrics, [])
    return flattened


def save_model_metrics(upload_id: int, task_type: str, metrics: Dict[str, Any]):
    entries = _flatten_metrics(metrics)
    if not entries:
        return

    conn = get_connection()
    cursor = conn.cursor()
    payload = [
        (upload_id, task_type, model_name, metric_key, metric_value, _utc_now())
        for model_name, metric_key, metric_value in entries
    ]
    cursor.executemany(
        """
        INSERT INTO model_metrics (upload_id, task_type, model_name, metric_key, metric_value, created_at)
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        payload,
    )
    conn.commit()
    cursor.close()
    conn.close()


def save_cluster_results(upload_id: int, task_type: str, clusters: List[Dict[str, Any]]):
    if not clusters:
        return

    conn = get_connection()
    cursor = conn.cursor()
    payload = [
        (
            upload_id,
            task_type,
            int(item.get("cluster", 0)),
            str(item.get("label", "Cluster")),
            int(item.get("count", 0)),
            _utc_now(),
        )
        for item in clusters
    ]
    cursor.executemany(
        """
        INSERT INTO cluster_results (upload_id, task_type, cluster_id, cluster_label, cluster_count, created_at)
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        payload,
    )
    conn.commit()
    cursor.close()
    conn.close()


def get_cluster_results(upload_id: int):
    conn = get_connection()
    cursor = _dict_cursor(conn)
    cursor.execute(
        """
        SELECT cluster_id AS cluster, cluster_label AS label, cluster_count AS count
        FROM cluster_results
        WHERE upload_id = %s
        ORDER BY cluster_id ASC
        """,
        (upload_id,),
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows


def save_prediction(
    user_id: Optional[int],
    upload_id: Optional[int],
    task_type: str,
    input_data: Dict[str, Any],
    output_value: Any,
    model_used: str,
    probability_score: Optional[float] = None,
    insight_text: Optional[str] = None,
):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO predictions (
            user_id, upload_id, task_type, input_json, output_value, model_used,
            probability_score, insight_text, created_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            user_id,
            upload_id,
            task_type,
            json.dumps(input_data),
            str(output_value),
            model_used,
            probability_score,
            insight_text,
            _utc_now(),
        ),
    )
    prediction_id = cursor.lastrowid
    conn.commit()
    cursor.close()
    conn.close()
    return prediction_id


def get_prediction_history(user_id: int, limit=100, upload_id: Optional[int] = None):
    conn = get_connection()
    cursor = _dict_cursor(conn)
    if upload_id is None:
        cursor.execute(
            """
            SELECT p.id, p.upload_id, p.task_type, p.input_json, p.output_value, p.model_used,
                   p.probability_score, p.insight_text, p.created_at,
                   COALESCE(u.filename, 'Ad-hoc Predictions') AS dataset_name
            FROM predictions p
            LEFT JOIN uploads u ON u.id = p.upload_id
            WHERE p.user_id = %s
            ORDER BY p.id DESC
            LIMIT %s
            """,
            (user_id, limit),
        )
    else:
        cursor.execute(
            """
            SELECT p.id, p.upload_id, p.task_type, p.input_json, p.output_value, p.model_used,
                   p.probability_score, p.insight_text, p.created_at,
                   COALESCE(u.filename, 'Ad-hoc Predictions') AS dataset_name
            FROM predictions p
            LEFT JOIN uploads u ON u.id = p.upload_id
            WHERE p.user_id = %s AND p.upload_id = %s
            ORDER BY p.id DESC
            LIMIT %s
            """,
            (user_id, upload_id, limit),
        )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows


def delete_prediction_by_id(prediction_id: int, user_id: int) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM predictions WHERE id = %s AND user_id = %s",
        (prediction_id, user_id),
    )
    deleted = int(cursor.rowcount)
    conn.commit()
    cursor.close()
    conn.close()
    return deleted


def delete_predictions_by_upload(upload_id: int, user_id: int) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM predictions WHERE upload_id = %s AND user_id = %s",
        (upload_id, user_id),
    )
    deleted = int(cursor.rowcount)
    conn.commit()
    cursor.close()
    conn.close()
    return deleted


def get_latest_upload(user_id: int):
    conn = get_connection()
    cursor = _dict_cursor(conn)
    cursor.execute(
        """
        SELECT id, filename, task_type, target_column, total_records, processed_records,
               metrics_json, feature_importance_json, insight_text, preview_json, created_at, storage_path
        FROM uploads
        WHERE user_id = %s
        ORDER BY id DESC
        LIMIT 1
        """,
        (user_id,),
    )
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    return row if row else None


def get_upload_history(user_id: int, limit: int = 20):
    conn = get_connection()
    cursor = _dict_cursor(conn)
    cursor.execute(
        """
        SELECT id, filename, task_type, target_column, total_records, processed_records, created_at, storage_path
        FROM uploads
        WHERE user_id = %s
        ORDER BY id DESC
        LIMIT %s
        """,
        (user_id, limit),
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows


def get_upload_by_id(upload_id: int, user_id: int):
    conn = get_connection()
    cursor = _dict_cursor(conn)
    cursor.execute(
        """
        SELECT id, filename, task_type, target_column, total_records, processed_records,
               metrics_json, feature_importance_json, insight_text, preview_json, created_at, storage_path
        FROM uploads
        WHERE id = %s AND user_id = %s
        LIMIT 1
        """,
        (upload_id, user_id),
    )
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    return row if row else None


def get_analytics_history(user_id: int, limit: int = 20):
    conn = get_connection()
    cursor = _dict_cursor(conn)
    cursor.execute(
        """
        SELECT
            u.id,
            u.filename,
            u.task_type,
            u.target_column,
            u.total_records,
            u.processed_records,
            u.metrics_json,
            u.created_at,
            COUNT(p.id) AS prediction_count,
            AVG(p.probability_score) AS avg_prediction_probability,
            MIN(p.created_at) AS first_prediction_at,
            MAX(p.created_at) AS last_prediction_at,
            GROUP_CONCAT(DISTINCT p.model_used ORDER BY p.model_used SEPARATOR ', ') AS prediction_models
        FROM uploads u
        LEFT JOIN predictions p ON p.upload_id = u.id
        WHERE u.user_id = %s
        GROUP BY u.id
        ORDER BY u.id DESC
        LIMIT %s
        """,
        (user_id, limit),
    )
    uploads = cursor.fetchall()

    if not uploads:
        cursor.close()
        conn.close()
        return []

    upload_ids = [int(item["id"]) for item in uploads]
    placeholders = ", ".join(["%s"] * len(upload_ids))
    cursor.execute(
        f"""
        SELECT upload_id, cluster_id AS cluster, cluster_label AS label, cluster_count AS count
        FROM cluster_results
        WHERE upload_id IN ({placeholders})
        ORDER BY upload_id DESC, cluster_id ASC
        """,
        tuple(upload_ids),
    )
    cluster_rows = cursor.fetchall()

    clusters_by_upload: Dict[int, List[Dict[str, Any]]] = {}
    for row in cluster_rows:
        upload_id = int(row["upload_id"])
        clusters_by_upload.setdefault(upload_id, []).append(
            {
                "cluster": int(row["cluster"]),
                "label": str(row["label"]),
                "count": int(row["count"]),
            }
        )

    results: List[Dict[str, Any]] = []
    for upload in uploads:
        metrics = json.loads(upload.get("metrics_json") or "{}")
        model_used = metrics.get("model_comparison", {}).get("winner") or "-"

        if upload["task_type"] == "classification":
            performance = metrics.get("classification", {}).get("random_forest", {})
        elif upload["task_type"] == "regression":
            performance = metrics.get("regression", {}).get("random_forest_regressor", {})
        else:
            performance = metrics.get("dashboard", {})

        results.append(
            {
                "id": int(upload["id"]),
                "dataset_name": upload["filename"],
                "task_type": upload["task_type"],
                "target_column": upload["target_column"],
                "training_date": upload["created_at"],
                "model_used": model_used,
                "performance_metrics": performance,
                "prediction_summary": {
                    "total_predictions": int(upload.get("prediction_count") or 0),
                    "average_probability": (
                        float(upload["avg_prediction_probability"])
                        if upload.get("avg_prediction_probability") is not None
                        else None
                    ),
                    "first_prediction_at": upload.get("first_prediction_at"),
                    "last_prediction_at": upload.get("last_prediction_at"),
                    "models": upload.get("prediction_models") or "",
                },
                "clustering_results": clusters_by_upload.get(int(upload["id"]), []),
                "records_summary": {
                    "processed_records": int(upload["processed_records"]),
                    "total_records": int(upload["total_records"]),
                },
            }
        )

    cursor.close()
    conn.close()
    return results


def delete_upload_and_related(upload_id: int, user_id: int) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = _dict_cursor(conn)
    cursor.execute(
        """
        SELECT id, filename, storage_path
        FROM uploads
        WHERE id = %s AND user_id = %s
        LIMIT 1
        """,
        (upload_id, user_id),
    )
    upload = cursor.fetchone()
    if not upload:
        cursor.close()
        conn.close()
        return None

    cursor.close()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions WHERE upload_id = %s", (upload_id,))
    deleted_predictions = int(cursor.rowcount)
    cursor.execute("DELETE FROM uploads WHERE id = %s AND user_id = %s", (upload_id, user_id))
    deleted_uploads = int(cursor.rowcount)
    conn.commit()
    cursor.close()
    conn.close()

    return {
        "id": int(upload["id"]),
        "filename": upload["filename"],
        "storage_path": upload.get("storage_path"),
        "deleted_uploads": deleted_uploads,
        "deleted_predictions": deleted_predictions,
    }
