from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class UserRow:
    id: str
    username: str
    password_hash: str
    role: str
    created_at: str


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK (role IN ('admin','petani')),
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS datasets (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            stored_path TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            feature_count INTEGER NOT NULL,
            target_column TEXT NOT NULL,
            uploaded_by TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS model_meta (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            model_path TEXT,
            trained_at TEXT,
            metrics_json TEXT,
            confusion_json TEXT,
            feature_importance_json TEXT,
            shap_summary_path TEXT,
            classes_json TEXT,
            feature_names_json TEXT,
            dataset_id TEXT
        );

        INSERT OR IGNORE INTO model_meta (id) VALUES (1);

        CREATE TABLE IF NOT EXISTS predictions (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            input_json TEXT NOT NULL,
            result_json TEXT NOT NULL,
            shap_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_predictions_user_created ON predictions(user_id, created_at);

        CREATE TABLE IF NOT EXISTS activity_logs (
            id TEXT PRIMARY KEY,
            actor_user_id TEXT,
            actor_username TEXT,
            actor_role TEXT,
            action TEXT NOT NULL,
            detail_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_activity_logs_created ON activity_logs(created_at);
        """
    )
    conn.commit()


def get_user_by_username(conn: sqlite3.Connection, username: str) -> Optional[UserRow]:
    row = conn.execute(
        "SELECT id, username, password_hash, role, created_at FROM users WHERE username = ?",
        (username,),
    ).fetchone()
    if not row:
        return None
    return UserRow(
        id=row["id"],
        username=row["username"],
        password_hash=row["password_hash"],
        role=row["role"],
        created_at=row["created_at"],
    )


def get_user_by_id(conn: sqlite3.Connection, user_id: str) -> Optional[UserRow]:
    row = conn.execute(
        "SELECT id, username, password_hash, role, created_at FROM users WHERE id = ?",
        (user_id,),
    ).fetchone()
    if not row:
        return None
    return UserRow(
        id=row["id"],
        username=row["username"],
        password_hash=row["password_hash"],
        role=row["role"],
        created_at=row["created_at"],
    )


def ensure_user(conn: sqlite3.Connection, username: str, password_hash: str, role: str) -> None:
    existing = conn.execute(
        "SELECT id FROM users WHERE username = ?",
        (username,),
    ).fetchone()
    if existing:
        return
    conn.execute(
        "INSERT INTO users (id, username, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?)",
        (str(uuid.uuid4()), username, password_hash, role, utc_now_iso()),
    )
    conn.commit()


def insert_dataset(
    conn: sqlite3.Connection,
    *,
    filename: str,
    stored_path: str,
    row_count: int,
    feature_count: int,
    target_column: str,
    uploaded_by: str,
) -> str:
    dataset_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO datasets (id, filename, stored_path, row_count, feature_count, target_column, uploaded_by, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (dataset_id, filename, stored_path, row_count, feature_count, target_column, uploaded_by, utc_now_iso()),
    )
    conn.commit()
    return dataset_id


def get_latest_dataset(conn: sqlite3.Connection) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM datasets ORDER BY created_at DESC LIMIT 1"
    ).fetchone()


def list_datasets(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(conn.execute("SELECT * FROM datasets ORDER BY created_at DESC").fetchall())


def get_dataset(conn: sqlite3.Connection, dataset_id: str) -> Optional[sqlite3.Row]:
    return conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()


def set_model_meta(
    conn: sqlite3.Connection,
    *,
    model_path: str,
    trained_at: str,
    metrics: dict[str, Any],
    confusion: dict[str, Any],
    feature_importance: dict[str, Any],
    shap_summary_path: str,
    classes: list[str],
    feature_names: list[str],
    dataset_id: str,
) -> None:
    conn.execute(
        """
        UPDATE model_meta
        SET model_path = ?,
            trained_at = ?,
            metrics_json = ?,
            confusion_json = ?,
            feature_importance_json = ?,
            shap_summary_path = ?,
            classes_json = ?,
            feature_names_json = ?,
            dataset_id = ?
        WHERE id = 1
        """,
        (
            model_path,
            trained_at,
            json.dumps(metrics, ensure_ascii=False),
            json.dumps(confusion, ensure_ascii=False),
            json.dumps(feature_importance, ensure_ascii=False),
            shap_summary_path,
            json.dumps(classes, ensure_ascii=False),
            json.dumps(feature_names, ensure_ascii=False),
            dataset_id,
        ),
    )
    conn.commit()


def get_model_meta(conn: sqlite3.Connection) -> sqlite3.Row:
    return conn.execute("SELECT * FROM model_meta WHERE id = 1").fetchone()


def insert_prediction(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    input_obj: dict[str, Any],
    result_obj: dict[str, Any],
    shap_obj: dict[str, Any],
) -> str:
    pred_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO predictions (id, user_id, input_json, result_json, shap_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            pred_id,
            user_id,
            json.dumps(input_obj, ensure_ascii=False),
            json.dumps(result_obj, ensure_ascii=False),
            json.dumps(shap_obj, ensure_ascii=False),
            utc_now_iso(),
        ),
    )
    conn.commit()
    return pred_id


def list_predictions(conn: sqlite3.Connection, user_id: str, limit: int = 50) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            "SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    )


def get_prediction(conn: sqlite3.Connection, pred_id: str, user_id: str) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM predictions WHERE id = ? AND user_id = ?",
        (pred_id, user_id),
    ).fetchone()


def insert_activity(
    conn: sqlite3.Connection,
    *,
    actor_user_id: Optional[str],
    actor_username: Optional[str],
    actor_role: Optional[str],
    action: str,
    detail: dict[str, Any],
) -> None:
    conn.execute(
        """
        INSERT INTO activity_logs (id, actor_user_id, actor_username, actor_role, action, detail_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid.uuid4()),
            actor_user_id,
            actor_username,
            actor_role,
            action,
            json.dumps(detail, ensure_ascii=False),
            utc_now_iso(),
        ),
    )
    conn.commit()


def list_activity(conn: sqlite3.Connection, limit: int = 200) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            "SELECT * FROM activity_logs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    )

