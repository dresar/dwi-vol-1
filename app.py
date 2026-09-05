from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    Body,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from joblib import dump, load
from markupsafe import Markup
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from starlette.middleware.sessions import SessionMiddleware

from utils import auth as auth_utils
from utils import db as db_utils
from utils import explainability as xai_utils
from utils import smart_monitor


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
STATIC_DIR = ROOT_DIR / "static"
PLOTS_DIR = STATIC_DIR / "plots"
LOGS_DIR = STATIC_DIR / "logs"
DB_PATH = DATA_DIR / "app.db"
MODEL_PATH = MODELS_DIR / "model.pkl"
SHAP_SUMMARY_PATH = PLOTS_DIR / "shap_summary.png"

APP_SECRET_KEY = os.environ.get("APP_SECRET_KEY", "dev-secret-change-me")


REQUIRED_FEATURES = [
    "temperature",
    "rainfall",
    "humidity",
    "ph",
    "nitrogen",
    "phosphorus",
    "potassium",
    "light_intensity",
    "soil_type",
]

TARGET_CANDIDATES = [
    "season",
    "musim_tanam",
    "recommended_season",
    "label",
    "target",
]

SOIL_TYPE_DEFAULTS = [
    "clay",
    "loam",
    "sandy",
    "silt",
    "peat",
    "laterite",
    "chalk",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


ensure_dirs()


logger = logging.getLogger("app")
logger.setLevel(logging.INFO)

_fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

_sh = logging.StreamHandler()
_sh.setLevel(logging.INFO)
_sh.setFormatter(_fmt)

_fh = logging.FileHandler(str(LOGS_DIR / "app.log"), encoding="utf-8")
_fh.setLevel(logging.INFO)
_fh.setFormatter(_fmt)

logger.handlers.clear()
logger.addHandler(_sh)
logger.addHandler(_fh)


app = FastAPI(title="Sistem Rekomendasi Musim Tanam", debug=False)
app.add_middleware(SessionMiddleware, secret_key=APP_SECRET_KEY, same_site="lax", https_only=False)
app.add_middleware(smart_monitor.SmartMiddleware)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


templates = Jinja2Templates(directory=str(ROOT_DIR / "templates"))


def _tojson(value: Any) -> Markup:
    return Markup(json.dumps(value, ensure_ascii=False))


templates.env.filters["tojson"] = _tojson


@dataclass
class TrainingRun:
    run_id: str
    status: str
    progress: int
    logs: list[str]
    started_at: float
    finished_at: Optional[float]
    error: Optional[str]
    step_state: dict[str, str]

    def elapsed_seconds(self) -> float:
        end = self.finished_at if self.finished_at is not None else time.time()
        return max(0.0, float(end - self.started_at))


TRAINING_LOCK = asyncio.Lock()
TRAINING_RUNS: dict[str, TrainingRun] = {}


def get_db() -> Any:
    return app.state.db


def get_session_user(request: Request) -> auth_utils.SessionUser:
    session_user = auth_utils.get_session_user(request.session)
    if not session_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return session_user


def require_role(role: str) -> Callable:
    def _dep(session_user: auth_utils.SessionUser = Depends(get_session_user)) -> auth_utils.SessionUser:
        if session_user.role != role:
            raise HTTPException(status_code=403, detail="Forbidden")
        return session_user

    return _dep


def render(request: Request, name: str, context: dict[str, Any], status_code: int = 200) -> HTMLResponse:
    session_user = auth_utils.get_session_user(request.session)
    flash = request.session.pop("flash", None)
    base = {
        "request": request,
        "session_user": session_user,
        "flash": flash,
        "year": datetime.now().year,
    }
    base.update(context)
    return templates.TemplateResponse(name, base, status_code=status_code)


def redirect(url: str, status_code: int = 303) -> RedirectResponse:
    return RedirectResponse(url=url, status_code=status_code)


def set_flash(request: Request, message: str) -> None:
    request.session["flash"] = message


def parse_float(value: str, name: str) -> float:
    try:
        v = float(value)
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid number for {name}")
    if not np.isfinite(v):
        raise HTTPException(status_code=400, detail=f"Invalid number for {name}")
    return v


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "pH": "ph",
        "PH": "ph",
        "ph_value": "ph",
        "ph_tanah": "ph",
        "temp": "temperature",
        "temperature_c": "temperature",
        "suhu": "temperature",
        "rain": "rainfall",
        "rainfall_mm": "rainfall",
        "curah_hujan": "rainfall",
        "curahhujan": "rainfall",
        "curah hujan": "rainfall",
        "hum": "humidity",
        "humid": "humidity",
        "kelembapan": "humidity",
        "kelembaban": "humidity",
        "rh": "humidity",
        "relative_humidity": "humidity",
        "relative humidity": "humidity",
        "relhum": "humidity",
        "n": "nitrogen",
        "p": "phosphorus",
        "fosfor": "phosphorus",
        "k": "potassium",
        "kalium": "potassium",
        "light": "light_intensity",
        "light_lux": "light_intensity",
        "intensitas_cahaya": "light_intensity",
        "intensitascahaya": "light_intensity",
        "intensitas cahaya": "light_intensity",
        "soil": "soil_type",
        "soiltype": "soil_type",
        "jenis_tanah": "soil_type",
        "jenistanah": "soil_type",
        "jenis tanah": "soil_type",
    }
    cols = {}
    for c in df.columns:
        c2 = str(c).strip()
        cols[c] = mapping.get(c2, mapping.get(c2.lower(), c2.lower()))
    df2 = df.rename(columns=cols)
    return df2


def find_target_column(df: pd.DataFrame) -> Optional[str]:
    cols = set([str(c).strip().lower() for c in df.columns])
    for cand in TARGET_CANDIDATES:
        if cand.lower() in cols:
            return cand.lower()
    for c in df.columns:
        if str(c).strip().lower() in TARGET_CANDIDATES:
            return str(c).strip().lower()
    return None


def validate_dataset(df: pd.DataFrame) -> tuple[list[str], str]:
    df = normalize_columns(df)
    cols = set(df.columns)
    missing = [c for c in REQUIRED_FEATURES if c not in cols]
    target = find_target_column(df)
    if not target:
        raise HTTPException(
            status_code=400,
            detail=f"Target column tidak ditemukan. Kandidat: {', '.join(TARGET_CANDIDATES)}",
        )
    if target not in cols:
        match = None
        for c in df.columns:
            if str(c).strip().lower() == target:
                match = c
                break
        if match:
            df = df.rename(columns={match: target})
        cols = set(df.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Kolom fitur wajib belum lengkap: {', '.join(missing)}",
        )
    return REQUIRED_FEATURES, target


def ensure_model_ready(conn: Any) -> dict[str, Any]:
    meta = db_utils.get_model_meta(conn)
    model_path = meta["model_path"]
    if not model_path or not MODEL_PATH.exists():
        raise HTTPException(status_code=400, detail="Model belum tersedia. Admin harus training terlebih dahulu.")
    return meta


def load_pipeline() -> Any:
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=400, detail="Model belum tersedia.")
    try:
        return load(str(MODEL_PATH))
    except Exception:
        raise HTTPException(status_code=500, detail="Gagal memuat model.")


def get_preprocessor_and_model(pipeline: Any) -> tuple[Any, Any]:
    if not hasattr(pipeline, "named_steps"):
        raise HTTPException(status_code=500, detail="Model format tidak valid.")
    pre = pipeline.named_steps.get("pre")
    clf = pipeline.named_steps.get("clf")
    if pre is None or clf is None:
        raise HTTPException(status_code=500, detail="Model pipeline tidak lengkap.")
    return pre, clf


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return pre


def build_pipeline() -> Pipeline:
    numeric_cols = [c for c in REQUIRED_FEATURES if c != "soil_type"]
    categorical_cols = ["soil_type"]
    pre = build_preprocessor(numeric_cols=numeric_cols, categorical_cols=categorical_cols)
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
    return pipe


def _step_name_to_percent(step: str) -> int:
    mapping = {
        "read": 8,
        "clean": 16,
        "encode": 24,
        "split": 32,
        "train": 64,
        "eval": 76,
        "cm": 84,
        "fi": 90,
        "shap": 96,
        "save": 100,
    }
    return int(mapping.get(step, 0))


async def training_update(run_id: str, *, log: Optional[str] = None, progress: Optional[int] = None) -> None:
    async with TRAINING_LOCK:
        run = TRAINING_RUNS.get(run_id)
        if not run:
            return
        if log:
            run.logs.append(log)
        if progress is not None:
            run.progress = int(max(0, min(100, progress)))


async def training_step(run_id: str, step: str, state: str) -> None:
    async with TRAINING_LOCK:
        run = TRAINING_RUNS.get(run_id)
        if not run:
            return
        run.step_state[step] = state
        if state == "on":
            run.progress = max(run.progress, _step_name_to_percent(step))
        if state == "done":
            run.progress = max(run.progress, _step_name_to_percent(step))


async def finalize_training(run_id: str, status: str, error: Optional[str] = None) -> None:
    async with TRAINING_LOCK:
        run = TRAINING_RUNS.get(run_id)
        if not run:
            return
        run.status = status
        run.finished_at = time.time()
        run.error = error
        if status == "done":
            run.progress = 100


def compute_feature_names(pre: Any) -> list[str]:
    try:
        names = pre.get_feature_names_out()
        return [str(n) for n in names]
    except Exception:
        return []


def compute_feature_importance(clf: Any, feature_names: list[str], top_k: int = 20) -> dict[str, Any]:
    try:
        importances = clf.feature_importances_.astype(float).tolist()
    except Exception:
        importances = []
    pairs = []
    for i, v in enumerate(importances):
        name = feature_names[i] if i < len(feature_names) else f"f{i}"
        pairs.append((name, float(v)))
    pairs.sort(key=lambda x: x[1], reverse=True)
    pairs = pairs[: max(5, min(top_k, len(pairs)))]
    return {"labels": [p[0] for p in pairs], "values": [p[1] for p in pairs]}


def save_shap_summary_plot(
    *,
    clf: Any,
    X_trans: np.ndarray,
    feature_names: list[str],
    out_path: Path,
    max_samples: int = 200,
) -> None:
    if X_trans.shape[0] == 0:
        return
    n = min(int(max_samples), int(X_trans.shape[0]))
    idx = np.random.RandomState(42).choice(X_trans.shape[0], size=n, replace=False)
    X_sample = X_trans[idx]
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_sample)
    plt.figure(figsize=(10, 6))
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[0], X_sample, feature_names=feature_names, show=False)
    else:
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=160)
    plt.close()


def compute_single_shap(
    *,
    clf: Any,
    x_trans_1: np.ndarray,
    feature_names: list[str],
    predicted_class_index: int,
    top_k: int = 12,
) -> dict[str, Any]:
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(x_trans_1)
    if isinstance(shap_values, list):
        if predicted_class_index < 0 or predicted_class_index >= len(shap_values):
            predicted_class_index = 0
        sv = shap_values[predicted_class_index][0]
    else:
        sv = shap_values[0]
    values = x_trans_1[0]
    top = xai_utils.top_contributions(feature_names=feature_names, values=values, shap_values_1d=sv, top_k=top_k)
    labels = [t["feature"] for t in top]
    vals = [float(t["shap_value"]) for t in top]
    return {
        "top": top,
        "chart": {"labels": labels, "values": vals},
    }


def safe_read_csv(file_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="latin-1")
    return df


def safe_read_tabular(file_path: Path) -> pd.DataFrame:
    p = Path(str(file_path))
    name = p.name.lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        try:
            df = pd.read_excel(p, engine="openpyxl")
            return df
        except Exception:
            df = pd.read_excel(p)
            return df
    return safe_read_csv(p)


def coerce_types(df: pd.DataFrame, features: list[str], target: str) -> pd.DataFrame:
    df = normalize_columns(df)
    for c in features:
        if c == "soil_type":
            df[c] = df[c].astype(str)
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df[target] = df[target].astype(str)
    return df


def build_confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    matrix = cm.astype(int).tolist()
    return {"labels": labels, "matrix": matrix}


def metric_pack(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": round(acc, 6),
        "precision": round(float(prec), 6),
        "recall": round(float(rec), 6),
        "f1": round(float(f1), 6),
    }


async def train_job(
    *,
    run_id: str,
    dataset_id: str,
    actor: Optional[auth_utils.SessionUser],
) -> None:
    conn = get_db()
    started = time.time()
    try:
        dataset = db_utils.get_dataset(conn, dataset_id)
        if not dataset:
            raise RuntimeError("Dataset tidak ditemukan")
        dataset_path = Path(str(dataset["stored_path"]))
        if not dataset_path.exists():
            raise RuntimeError("File dataset tidak ditemukan")

        await training_step(run_id, "read", "on")
        await training_update(run_id, log="Membaca dataset...", progress=_step_name_to_percent("read"))
        df = await asyncio.to_thread(safe_read_csv, dataset_path)
        df = normalize_columns(df)
        features, target = validate_dataset(df)
        df = coerce_types(df, features, target)
        await training_step(run_id, "read", "done")

        await training_step(run_id, "clean", "on")
        await training_update(run_id, log="Membersihkan missing value dan validasi data...", progress=_step_name_to_percent("clean"))
        df = df[features + [target]].copy()
        df = df.dropna(subset=[target])
        for c in features:
            if c == "soil_type":
                df[c] = df[c].fillna("unknown")
            else:
                med = df[c].median()
                df[c] = df[c].fillna(med)
        await training_step(run_id, "clean", "done")

        await training_step(run_id, "encode", "on")
        await training_update(run_id, log="Menyiapkan preprocessing (encoding kategorikal)...", progress=_step_name_to_percent("encode"))
        X = df[features]
        y = df[target].astype(str)
        await training_step(run_id, "encode", "done")

        await training_step(run_id, "split", "on")
        await training_update(run_id, log="Splitting data train/test 80:20...", progress=_step_name_to_percent("split"))
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y if y.nunique() > 1 else None,
        )
        await training_step(run_id, "split", "done")

        await training_step(run_id, "train", "on")
        await training_update(run_id, log="Training RandomForestClassifier (n_estimators=300)...", progress=_step_name_to_percent("train"))
        pipe = build_pipeline()
        pipe = await asyncio.to_thread(pipe.fit, X_train, y_train)
        await training_step(run_id, "train", "done")

        await training_step(run_id, "eval", "on")
        await training_update(run_id, log="Evaluasi model (accuracy, precision, recall, f1)...", progress=_step_name_to_percent("eval"))
        y_pred = await asyncio.to_thread(pipe.predict, X_test)
        metrics = metric_pack(y_test.to_numpy(), np.asarray(y_pred))
        await training_step(run_id, "eval", "done")

        pre, clf = get_preprocessor_and_model(pipe)
        feature_names = compute_feature_names(pre)
        classes = [str(c) for c in getattr(clf, "classes_", [])]
        if not classes:
            classes = sorted(list(pd.Series(y).unique().astype(str)))

        await training_step(run_id, "cm", "on")
        await training_update(run_id, log="Membuat confusion matrix...", progress=_step_name_to_percent("cm"))
        confusion = build_confusion(y_test.to_numpy().astype(str), np.asarray(y_pred).astype(str), labels=classes)
        await training_step(run_id, "cm", "done")

        await training_step(run_id, "fi", "on")
        await training_update(run_id, log="Menghitung feature importance...", progress=_step_name_to_percent("fi"))
        feature_importance = compute_feature_importance(clf, feature_names)
        await training_step(run_id, "fi", "done")

        await training_step(run_id, "shap", "on")
        await training_update(run_id, log="Menjalankan SHAP summary plot...", progress=_step_name_to_percent("shap"))
        X_train_trans = await asyncio.to_thread(pre.transform, X_train)
        if hasattr(X_train_trans, "toarray"):
            X_train_trans = X_train_trans.toarray()
        X_train_trans = np.asarray(X_train_trans)
        await asyncio.to_thread(
            save_shap_summary_plot,
            clf=clf,
            X_trans=X_train_trans,
            feature_names=feature_names,
            out_path=SHAP_SUMMARY_PATH,
        )
        await training_step(run_id, "shap", "done")

        await training_step(run_id, "save", "on")
        await training_update(run_id, log="Menyimpan model ke models/model.pkl...", progress=_step_name_to_percent("save"))
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(dump, pipe, str(MODEL_PATH))
        trained_at = utc_now_iso()
        db_utils.set_model_meta(
            conn,
            model_path=str(MODEL_PATH),
            trained_at=trained_at,
            metrics=metrics,
            confusion=confusion,
            feature_importance=feature_importance,
            shap_summary_path=str(SHAP_SUMMARY_PATH),
            classes=classes,
            feature_names=feature_names,
            dataset_id=dataset_id,
        )
        await training_step(run_id, "save", "done")

        duration = time.time() - started
        await training_update(run_id, log=f"Training selesai. Total waktu: {duration:.2f} detik", progress=100)
        db_utils.insert_activity(
            conn,
            actor_user_id=actor.id if actor else None,
            actor_username=actor.username if actor else None,
            actor_role=actor.role if actor else None,
            action="TRAIN_MODEL",
            detail={
                "dataset_id": dataset_id,
                "model_path": str(MODEL_PATH),
                "metrics": metrics,
                "duration_seconds": round(duration, 3),
            },
        )
        await finalize_training(run_id, "done")
        app.state.last_training_run_id = run_id
    except Exception as e:
        err = str(e)
        logger.error("Training error: %s", err)
        logger.error(traceback.format_exc())
        await training_update(run_id, log=f"ERROR: {err}")
        db_utils.insert_activity(
            conn,
            actor_user_id=actor.id if actor else None,
            actor_username=actor.username if actor else None,
            actor_role=actor.role if actor else None,
            action="TRAIN_MODEL_ERROR",
            detail={"error": err, "trace": traceback.format_exc()[-1800:]},
        )
        await finalize_training(run_id, "error", error=err)


async def sse_training_events(run_id: str) -> AsyncIterator[bytes]:
    last_idx = 0
    while True:
        await asyncio.sleep(0.35)
        async with TRAINING_LOCK:
            run = TRAINING_RUNS.get(run_id)
            if not run:
                payload = {"status": "error", "error": "run_not_found"}
                yield f"data: {json.dumps(payload)}\n\n".encode("utf-8")
                return
            new_logs = run.logs[last_idx:]
            last_idx = len(run.logs)
            status = run.status
            progress = run.progress
            elapsed = run.elapsed_seconds()
            step_state = run.step_state.copy()
            error = run.error

        if new_logs:
            for line in new_logs:
                payload = {
                    "status": status,
                    "progress": progress,
                    "log": line,
                    "elapsed_seconds": elapsed,
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")
        else:
            payload = {
                "status": status,
                "progress": progress,
                "elapsed_seconds": elapsed,
            }
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")

        if step_state:
            for name, st in step_state.items():
                payload = {
                    "status": status,
                    "progress": progress,
                    "step": {"name": name, "state": st},
                    "elapsed_seconds": elapsed,
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")

        if status in ("done", "error"):
            payload = {
                "status": status,
                "progress": progress,
                "elapsed_seconds": elapsed,
                "error": error,
            }
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")
            return


@app.exception_handler(HTTPException)
async def http_exc_handler(request: Request, exc: HTTPException) -> Response:
    wants_json = request.url.path.startswith("/api/")
    if wants_json:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    if exc.status_code in (401, 403):
        if exc.status_code == 401:
            return redirect("/login")
        return render(
            request,
            "error.html",
            {"title": "Akses Ditolak", "message": "Halaman ini tidak bisa diakses untuk role kamu."},
            status_code=403,
        )
    return render(
        request,
        "error.html",
        {"title": f"Error {exc.status_code}", "message": str(exc.detail)},
        status_code=exc.status_code,
    )


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError) -> Response:
    wants_json = request.url.path.startswith("/api/")
    if wants_json:
        return JSONResponse(status_code=422, content={"detail": exc.errors()})
    return render(
        request,
        "error.html",
        {"title": "Validasi Gagal", "message": "Input tidak valid. Periksa kembali form."},
        status_code=422,
    )


@app.on_event("startup")
async def on_startup() -> None:
    conn = db_utils.connect(DB_PATH)
    db_utils.init_db(conn)
    app.state.db = conn
    app.state.last_training_run_id = None

    admin_hash = auth_utils.hash_password("admin123")
    petani_hash = auth_utils.hash_password("petani123")
    db_utils.ensure_user(conn, "admin", admin_hash, "admin")
    db_utils.ensure_user(conn, "petani", petani_hash, "petani")
    db_utils.insert_activity(
        conn,
        actor_user_id=None,
        actor_username=None,
        actor_role=None,
        action="STARTUP",
        detail={"message": "App started and dummy users ensured"},
    )
    logger.info("Startup complete")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request) -> Response:
    session_user = auth_utils.get_session_user(request.session)
    if session_user:
        return redirect("/dashboard")
    return redirect("/login")


@app.get("/login", response_class=HTMLResponse)
async def login_get(request: Request) -> Response:
    session_user = auth_utils.get_session_user(request.session)
    if session_user:
        return redirect("/dashboard")
    return templates.TemplateResponse(
        "login.html",
        {"request": request, "error": None, "username": "", "year": datetime.now().year},
    )


@app.post("/login", response_class=HTMLResponse)
async def login_post(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
) -> Response:
    conn = get_db()
    username_clean = str(username).strip()
    if not username_clean or len(username_clean) > 64:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Username tidak valid", "username": username_clean, "year": datetime.now().year},
            status_code=400,
        )
    user = db_utils.get_user_by_username(conn, username_clean)
    if not user:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Username atau password salah", "username": username_clean, "year": datetime.now().year},
            status_code=401,
        )
    ok = auth_utils.verify_password(password, user.password_hash)
    if not ok:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Username atau password salah", "username": username_clean, "year": datetime.now().year},
            status_code=401,
        )
    request.session.clear()
    request.session["user_id"] = user.id
    request.session["username"] = user.username
    request.session["role"] = user.role
    db_utils.insert_activity(
        conn,
        actor_user_id=user.id,
        actor_username=user.username,
        actor_role=user.role,
        action="LOGIN",
        detail={"username": user.username},
    )
    set_flash(request, "Login berhasil")
    return redirect("/dashboard")


@app.get("/logout")
async def logout(request: Request) -> Response:
    conn = get_db()
    session_user = auth_utils.get_session_user(request.session)
    if session_user:
        db_utils.insert_activity(
            conn,
            actor_user_id=session_user.id,
            actor_username=session_user.username,
            actor_role=session_user.role,
            action="LOGOUT",
            detail={},
        )
    request.session.clear()
    return redirect("/login")


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, session_user: auth_utils.SessionUser = Depends(get_session_user)) -> Response:
    conn = get_db()
    meta = db_utils.get_model_meta(conn)
    model_status = "trained" if meta["model_path"] and MODEL_PATH.exists() else "belum"
    metrics = {}
    if meta["metrics_json"]:
        try:
            metrics = json.loads(meta["metrics_json"])
        except Exception:
            metrics = {}
    latest = db_utils.get_latest_dataset(conn)
    total_datasets = conn.execute("SELECT COUNT(*) AS c FROM datasets").fetchone()["c"]
    latest_feature_count = int(latest["feature_count"]) if latest else 0
    latest_accuracy = metrics.get("accuracy") if metrics else None
    latest_accuracy_str = f"{float(latest_accuracy):.4f}" if latest_accuracy is not None else "—"

    confusion = {}
    feature_importance = {}
    if meta["confusion_json"]:
        try:
            confusion = json.loads(meta["confusion_json"])
        except Exception:
            confusion = {}
    if meta["feature_importance_json"]:
        try:
            feature_importance = json.loads(meta["feature_importance_json"])
        except Exception:
            feature_importance = {}

    if session_user.role == "admin":
        stats = {
            "total_datasets": int(total_datasets),
            "latest_feature_count": latest_feature_count,
            "latest_accuracy": latest_accuracy_str,
            "model_status": model_status,
        }
        charts = {"confusion": confusion, "feature_importance": feature_importance}
        return render(
            request,
            "dashboard_admin.html",
            {
                "title": "Dashboard Admin",
                "page_title": "Dashboard Admin",
                "page_subtitle": "Ringkasan sistem, status model, dan evaluasi",
                "active": "dashboard",
                "stats": stats,
                "charts_json": json.dumps(charts, ensure_ascii=False),
            },
        )

    recent_rows = db_utils.list_predictions(conn, session_user.id, limit=8)
    recent = []
    last_pred = None
    for r in recent_rows:
        try:
            res = json.loads(r["result_json"])
            label = res.get("label")
        except Exception:
            label = "-"
        item = {"id": r["id"], "label": label, "created_at": r["created_at"]}
        recent.append(item)
    if recent:
        last_pred = {"label": recent[0]["label"], "created_at": recent[0]["created_at"]}
    stats2 = {"model_status": model_status}
    return render(
        request,
        "dashboard_petani.html",
        {
            "title": "Dashboard Petani",
            "page_title": "Dashboard Petani",
            "page_subtitle": "Akses cepat ke prediksi dan riwayat",
            "active": "dashboard",
            "stats": stats2,
            "recent": recent,
            "last_pred": last_pred,
        },
    )


@app.get("/admin/upload-dataset", response_class=HTMLResponse)
async def admin_upload_get(
    request: Request,
    session_user: auth_utils.SessionUser = Depends(require_role("admin")),
) -> Response:
    conn = get_db()
    latest = db_utils.get_latest_dataset(conn)
    latest_ctx = None
    if latest:
        latest_ctx = {
            "filename": latest["filename"],
            "stored_path": latest["stored_path"],
            "row_count": latest["row_count"],
            "feature_count": latest["feature_count"],
            "target_column": latest["target_column"],
            "created_at": latest["created_at"],
        }
    return render(
        request,
        "upload_dataset.html",
        {
            "title": "Upload Dataset",
            "page_title": "Upload Dataset",
            "page_subtitle": "Unggah CSV untuk training model",
            "active": "upload",
            "required_columns": REQUIRED_FEATURES + ["(target: season/musim_tanam/label/target)"] ,
            "latest": latest_ctx,
            "error": None,
            "ok": None,
        },
    )


@app.post("/admin/upload-dataset", response_class=HTMLResponse)
async def admin_upload_post(
    request: Request,
    session_user: auth_utils.SessionUser = Depends(require_role("admin")),
    file: UploadFile = File(...),
) -> Response:
    conn = get_db()
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Nama file tidak valid")
        if not file.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="File harus CSV")
        raw = await file.read()
        if len(raw) < 10:
            raise HTTPException(status_code=400, detail="File kosong")
        if len(raw) > 25 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File terlalu besar (maks 25MB)")

        dataset_name = f"dataset_{uuid.uuid4().hex}.csv"
        stored_path = DATA_DIR / dataset_name
        stored_path.write_bytes(raw)

        df = safe_read_csv(stored_path)
        df = normalize_columns(df)
        features, target = validate_dataset(df)
        df = coerce_types(df, features, target)
        df = df[features + [target]].copy()
        df = df.dropna(subset=[target])
        row_count = int(len(df))
        feature_count = int(len(features))
        if row_count < 10:
            raise HTTPException(status_code=400, detail="Dataset terlalu kecil. Minimal 10 baris.")
        if df[target].nunique() < 2:
            raise HTTPException(status_code=400, detail="Target harus memiliki minimal 2 kelas.")

        dataset_id = db_utils.insert_dataset(
            conn,
            filename=file.filename,
            stored_path=str(stored_path),
            row_count=row_count,
            feature_count=feature_count,
            target_column=target,
            uploaded_by=session_user.username,
        )

        db_utils.insert_activity(
            conn,
            actor_user_id=session_user.id,
            actor_username=session_user.username,
            actor_role=session_user.role,
            action="UPLOAD_DATASET",
            detail={
                "dataset_id": dataset_id,
                "filename": file.filename,
                "stored_path": str(stored_path),
                "row_count": row_count,
                "feature_count": feature_count,
                "target": target,
            },
        )
        set_flash(request, "Dataset berhasil diunggah")
        return redirect("/admin/upload-dataset")
    except HTTPException as e:
        latest = db_utils.get_latest_dataset(conn)
        latest_ctx = None
        if latest:
            latest_ctx = {
                "filename": latest["filename"],
                "stored_path": latest["stored_path"],
                "row_count": latest["row_count"],
                "feature_count": latest["feature_count"],
                "target_column": latest["target_column"],
                "created_at": latest["created_at"],
            }
        return render(
            request,
            "upload_dataset.html",
            {
                "title": "Upload Dataset",
                "page_title": "Upload Dataset",
                "page_subtitle": "Unggah CSV untuk training model",
                "active": "upload",
                "required_columns": REQUIRED_FEATURES + ["(target: season/musim_tanam/label/target)"] ,
                "latest": latest_ctx,
                "error": str(e.detail),
                "ok": None,
            },
            status_code=e.status_code,
        )


@app.get("/admin/training", response_class=HTMLResponse)
async def admin_training_page(
    request: Request,
    session_user: auth_utils.SessionUser = Depends(require_role("admin")),
) -> Response:
    conn = get_db()
    latest = db_utils.get_latest_dataset(conn)
    can_train = latest is not None
    last_run_id = getattr(app.state, "last_training_run_id", None)
    return render(
        request,
        "training_progress.html",
        {
            "title": "Training Model",
            "page_title": "Training Model",
            "page_subtitle": "Progress real-time pelatihan Random Forest",
            "active": "training",
            "can_train": can_train,
            "last_run_id": last_run_id,
        },
    )


@app.post("/admin/training/start")
async def admin_training_start(
    request: Request,
    session_user: auth_utils.SessionUser = Depends(require_role("admin")),
) -> Response:
    conn = get_db()
    latest = db_utils.get_latest_dataset(conn)
    if not latest:
        raise HTTPException(status_code=400, detail="Dataset belum tersedia")
    run_id = uuid.uuid4().hex
    run = TrainingRun(
        run_id=run_id,
        status="running",
        progress=0,
        logs=["Inisialisasi training"],
        started_at=time.time(),
        finished_at=None,
        error=None,
        step_state={},
    )
    async with TRAINING_LOCK:
        TRAINING_RUNS[run_id] = run
    app.state.last_training_run_id = run_id

    asyncio.create_task(train_job(run_id=run_id, dataset_id=str(latest["id"]), actor=session_user))
    return JSONResponse({"run_id": run_id})


@app.get("/admin/training/stream/{run_id}")
async def admin_training_stream(
    run_id: str,
    request: Request,
    session_user: auth_utils.SessionUser = Depends(require_role("admin")),
) -> Response:
    async def event_gen() -> AsyncIterator[bytes]:
        async for ev in sse_training_events(run_id):
            if await request.is_disconnected():
                return
            yield ev

    return StreamingResponse(event_gen(), media_type="text/event-stream")


def _format_metric(v: Any) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.4f}"
    except Exception:
        return str(v)


@app.get("/admin/evaluasi", response_class=HTMLResponse)
async def admin_evaluasi_page(
    request: Request,
    session_user: auth_utils.SessionUser = Depends(require_role("admin")),
) -> Response:
    conn = get_db()
    meta = db_utils.get_model_meta(conn)
    metrics_obj = {"accuracy": "—", "precision": "—", "recall": "—", "f1": "—"}
    confusion = {"labels": [], "matrix": []}
    feature_importance = {"labels": [], "values": []}
    trained_at = None
    shap_summary_url = None

    if meta["trained_at"] and meta["model_path"] and MODEL_PATH.exists():
        trained_at = meta["trained_at"]
        if meta["metrics_json"]:
            try:
                m = json.loads(meta["metrics_json"])
                metrics_obj = {
                    "accuracy": _format_metric(m.get("accuracy")),
                    "precision": _format_metric(m.get("precision")),
                    "recall": _format_metric(m.get("recall")),
                    "f1": _format_metric(m.get("f1")),
                }
            except Exception:
                pass
        if meta["confusion_json"]:
            try:
                confusion = json.loads(meta["confusion_json"])
            except Exception:
                pass
        if meta["feature_importance_json"]:
            try:
                feature_importance = json.loads(meta["feature_importance_json"])
            except Exception:
                pass
        if meta["shap_summary_path"]:
            p = Path(str(meta["shap_summary_path"]))
            if p.exists():
                shap_summary_url = "/static/plots/" + p.name

    payload = {"confusion": confusion, "feature_importance": feature_importance}
    return render(
        request,
        "evaluasi_model.html",
        {
            "title": "Evaluasi Model",
            "page_title": "Evaluasi Model",
            "page_subtitle": "Confusion matrix, metrik, feature importance, dan SHAP plot",
            "active": "evaluasi",
            "metrics": metrics_obj,
            "trained_at": trained_at,
            "shap_summary_url": shap_summary_url,
            "eval_json": json.dumps(payload, ensure_ascii=False),
        },
    )


@app.get("/admin/transparansi", response_class=HTMLResponse)
async def admin_transparansi_page(
    request: Request,
    session_user: auth_utils.SessionUser = Depends(require_role("admin")),
) -> Response:
    conn = get_db()
    meta = db_utils.get_model_meta(conn)
    feature_importance = {"labels": [], "values": []}
    shap_summary_url = None
    if meta["feature_importance_json"]:
        try:
            feature_importance = json.loads(meta["feature_importance_json"])
        except Exception:
            feature_importance = {"labels": [], "values": []}
    if meta["shap_summary_path"]:
        p = Path(str(meta["shap_summary_path"]))
        if p.exists():
            shap_summary_url = "/static/plots/" + p.name
    payload = {"feature_importance": feature_importance}
    return render(
        request,
        "transparansi_model.html",
        {
            "title": "Transparansi Model",
            "page_title": "Transparansi Model",
            "page_subtitle": "Interpretasi global model berbasis SHAP",
            "active": "transparansi",
            "shap_summary_url": shap_summary_url,
            "transparency_json": json.dumps(payload, ensure_ascii=False),
        },
    )


@app.get("/admin/log-aktivitas", response_class=HTMLResponse)
async def admin_log_aktivitas(
    request: Request,
    session_user: auth_utils.SessionUser = Depends(require_role("admin")),
) -> Response:
    conn = get_db()
    rows = db_utils.list_activity(conn, limit=200)
    return render(
        request,
        "log_aktivitas.html",
        {
            "title": "Log Aktivitas",
            "page_title": "Log Aktivitas",
            "page_subtitle": "Audit ringkas untuk Admin",
            "active": "log",
            "rows": rows,
        },
    )


def build_petani_fields() -> list[dict[str, Any]]:
    return [
        {"name": "temperature", "label": "Temperature (°C)", "type": "number", "placeholder": "contoh: 27.5"},
        {"name": "rainfall", "label": "Rainfall (mm)", "type": "number", "placeholder": "contoh: 210"},
        {"name": "humidity", "label": "Humidity (%)", "type": "number", "placeholder": "contoh: 78"},
        {"name": "ph", "label": "pH", "type": "number", "placeholder": "contoh: 6.2"},
        {"name": "nitrogen", "label": "Nitrogen (N)", "type": "number", "placeholder": "contoh: 35"},
        {"name": "phosphorus", "label": "Phosphorus (P)", "type": "number", "placeholder": "contoh: 20"},
        {"name": "potassium", "label": "Potassium (K)", "type": "number", "placeholder": "contoh: 18"},
        {"name": "light_intensity", "label": "Light Intensity", "type": "number", "placeholder": "contoh: 12000"},
        {"name": "soil_type", "label": "Soil Type", "type": "select", "options": SOIL_TYPE_DEFAULTS},
    ]


def validate_petani_input(form: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for c in REQUIRED_FEATURES:
        if c == "soil_type":
            v = str(form.get(c, "")).strip()
            if not v:
                raise HTTPException(status_code=400, detail="Soil type wajib diisi")
            out[c] = v
        else:
            v = form.get(c)
            if v is None or str(v).strip() == "":
                raise HTTPException(status_code=400, detail=f"{c} wajib diisi")
            f = parse_float(str(v), c)
            out[c] = round(float(f), 6)
            
    anomalies = smart_monitor.AnomalyDetector.detect_anomalies(out)
    if anomalies:
        raise HTTPException(status_code=400, detail=f"Validasi Cerdas: {'; '.join(anomalies)}")
        
    return out


def do_predict(*, conn: Any, input_obj: dict[str, Any]) -> dict[str, Any]:
    meta = ensure_model_ready(conn)
    t0 = time.perf_counter()
    pipe = load_pipeline()
    pre, clf = get_preprocessor_and_model(pipe)
    df = pd.DataFrame([input_obj], columns=REQUIRED_FEATURES)
    proba = pipe.predict_proba(df)
    pred = pipe.predict(df)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    labels = [str(c) for c in getattr(clf, "classes_", [])]
    if not labels:
        labels = json.loads(meta["classes_json"]) if meta["classes_json"] else []
    pred_label = str(pred[0])
    pred_idx = 0
    if labels and pred_label in labels:
        pred_idx = labels.index(pred_label)

    x_trans = pre.transform(df)
    if hasattr(x_trans, "toarray"):
        x_trans = x_trans.toarray()
    x_trans = np.asarray(x_trans)

    feature_names = compute_feature_names(pre)
    if not feature_names:
        if meta["feature_names_json"]:
            try:
                feature_names = json.loads(meta["feature_names_json"])
            except Exception:
                feature_names = []

    xai = compute_single_shap(
        clf=clf,
        x_trans_1=x_trans,
        feature_names=feature_names,
        predicted_class_index=pred_idx,
        top_k=12,
    )

    probs = proba[0].astype(float).tolist() if hasattr(proba, "__getitem__") else []
    if labels and len(labels) == len(probs):
        prob_labels = labels
    else:
        prob_labels = [f"kelas_{i}" for i in range(len(probs))]

    prob_payload = {"labels": prob_labels, "values": probs}
    out = {
        "label": pred_label,
        "latency_ms": round(float(latency_ms), 3),
        "probabilities": prob_payload,
        "top_contrib": xai["top"],
        "contributions": xai["chart"],
        "model_trained_at": meta["trained_at"],
    }
    shap_obj = {
        "top_contrib": xai["top"],
        "chart": xai["chart"],
        "predicted_class": pred_label,
    }
    return {"result": out, "shap": shap_obj}


@app.get("/petani/prediksi", response_class=HTMLResponse)
async def petani_prediksi_get(
    request: Request,
    session_user: auth_utils.SessionUser = Depends(require_role("petani")),
) -> Response:
    return render(
        request,
        "prediksi_musim_tanam.html",
        {
            "title": "Prediksi Musim Tanam",
            "page_title": "Prediksi Musim Tanam",
            "page_subtitle": "Input parameter untuk rekomendasi musim tanam",
            "active": "prediksi",
            "fields": build_petani_fields(),
            "error": None,
            "values": None,
        },
    )


@app.post("/petani/prediksi", response_class=HTMLResponse)
async def petani_prediksi_post(
    request: Request,
    session_user: auth_utils.SessionUser = Depends(require_role("petani")),
    temperature: str = Form(...),
    rainfall: str = Form(...),
    humidity: str = Form(...),
    ph: str = Form(...),
    nitrogen: str = Form(...),
    phosphorus: str = Form(...),
    potassium: str = Form(...),
    light_intensity: str = Form(...),
    soil_type: str = Form(...),
) -> Response:
    conn = get_db()
    form = {
        "temperature": temperature,
        "rainfall": rainfall,
        "humidity": humidity,
        "ph": ph,
        "nitrogen": nitrogen,
        "phosphorus": phosphorus,
        "potassium": potassium,
        "light_intensity": light_intensity,
        "soil_type": soil_type,
    }
    try:
        input_obj = validate_petani_input(form)
        pred = await asyncio.to_thread(do_predict, conn=conn, input_obj=input_obj)
        result_obj = pred["result"]
        shap_obj = pred["shap"]
        pred_id = db_utils.insert_prediction(
            conn,
            user_id=session_user.id,
            input_obj=input_obj,
            result_obj=result_obj,
            shap_obj=shap_obj,
        )
        db_utils.insert_activity(
            conn,
            actor_user_id=session_user.id,
            actor_username=session_user.username,
            actor_role=session_user.role,
            action="PREDICT",
            detail={"prediction_id": pred_id, "label": result_obj.get("label")},
        )
        set_flash(request, "Prediksi berhasil")
        return redirect(f"/petani/result/{pred_id}")
    except HTTPException as e:
        return render(
            request,
            "prediksi_musim_tanam.html",
            {
                "title": "Prediksi Musim Tanam",
                "page_title": "Prediksi Musim Tanam",
                "page_subtitle": "Input parameter untuk rekomendasi musim tanam",
                "active": "prediksi",
                "fields": build_petani_fields(),
                "error": str(e.detail),
                "values": form,
            },
            status_code=e.status_code,
        )


@app.get("/petani/riwayat", response_class=HTMLResponse)
async def petani_riwayat(
    request: Request,
    session_user: auth_utils.SessionUser = Depends(require_role("petani")),
) -> Response:
    conn = get_db()
    rows = db_utils.list_predictions(conn, session_user.id, limit=80)
    ctx_rows = []
    for r in rows:
        try:
            res = json.loads(r["result_json"])
            label = res.get("label")
        except Exception:
            label = "-"
        ctx_rows.append({"id": r["id"], "created_at": r["created_at"], "label": label})
    return render(
        request,
        "riwayat_prediksi.html",
        {
            "title": "Riwayat Prediksi",
            "page_title": "Riwayat Prediksi",
            "page_subtitle": "Semua prediksi yang pernah kamu lakukan",
            "active": "riwayat",
            "rows": ctx_rows,
        },
    )


@app.get("/petani/result/{pred_id}", response_class=HTMLResponse)
async def petani_result(
    request: Request,
    pred_id: str,
    session_user: auth_utils.SessionUser = Depends(require_role("petani")),
) -> Response:
    conn = get_db()
    row = db_utils.get_prediction(conn, pred_id, session_user.id)
    if not row:
        raise HTTPException(status_code=404, detail="Data tidak ditemukan")
    input_obj = {}
    result_obj = {}
    shap_obj = {}
    try:
        input_obj = json.loads(row["input_json"])
    except Exception:
        input_obj = {}
    try:
        result_obj = json.loads(row["result_json"])
    except Exception:
        result_obj = {}
    try:
        shap_obj = json.loads(row["shap_json"])
    except Exception:
        shap_obj = {}

    payload = {
        "probabilities": result_obj.get("probabilities", {"labels": [], "values": []}),
        "contributions": result_obj.get("contributions", {"labels": [], "values": []}),
    }
    out = {
        "id": row["id"],
        "created_at": row["created_at"],
        "label": result_obj.get("label", "-"),
        "latency_ms": result_obj.get("latency_ms", "-"),
        "input": input_obj,
        "top_contrib": result_obj.get("top_contrib", shap_obj.get("top_contrib", [])),
    }
    return render(
        request,
        "result.html",
        {
            "title": "Hasil Prediksi",
            "page_title": "Hasil Prediksi",
            "page_subtitle": "Rekomendasi dan transparansi keputusan",
            "active": "prediksi",
            "result": out,
            "result_json": json.dumps({**payload, **{"label": out["label"], "latency_ms": out["latency_ms"], "input": out["input"], "top_contrib": out["top_contrib"]}}, ensure_ascii=False),
        },
    )


@app.post("/api/predict")
async def api_predict(payload: dict[str, Any], request: Request) -> Response:
    conn = get_db()
    session_user = auth_utils.get_session_user(request.session)
    if not session_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    input_obj = validate_petani_input(payload)
    pred = await asyncio.to_thread(do_predict, conn=conn, input_obj=input_obj)
    return JSONResponse(pred["result"])


@app.get("/api/admin/model")
async def api_admin_model(request: Request, session_user: auth_utils.SessionUser = Depends(require_role("admin"))) -> Response:
    conn = get_db()
    meta = db_utils.get_model_meta(conn)
    out = {
        "model_path": meta["model_path"],
        "trained_at": meta["trained_at"],
        "metrics": json.loads(meta["metrics_json"]) if meta["metrics_json"] else None,
        "dataset_id": meta["dataset_id"],
    }
    return JSONResponse(out)


@app.get("/health")
async def health() -> Response:
    conn = get_db()
    _ = conn.execute("SELECT 1").fetchone()
    status = await asyncio.to_thread(smart_monitor.SystemDiagnostics.get_system_health)
    status["ok"] = True
    status["db_connected"] = True
    return JSONResponse(status)


@app.post("/admin/import-local")
async def admin_import_local(
    request: Request,
    session_user: auth_utils.SessionUser = Depends(require_role("admin")),
    payload: Optional[dict[str, Any]] = Body(None),
) -> Response:
    conn = get_db()
    path_str = ""
    if payload and "path" in payload:
        path_str = str(payload.get("path", "")).strip()
    if not path_str:
        path_str = str(request.query_params.get("path", "")).strip()
    if not path_str:
        raise HTTPException(status_code=400, detail="Path file wajib diisi")
    src_path = Path(path_str)
    if not src_path.exists() or not src_path.is_file():
        raise HTTPException(status_code=404, detail="File tidak ditemukan")
    df = await asyncio.to_thread(safe_read_tabular, src_path)
    df = normalize_columns(df)
    features, target = validate_dataset(df)
    df = coerce_types(df, features, target)
    df = df[features + [target]].copy()
    df = df.dropna(subset=[target])
    row_count = int(len(df))
    feature_count = int(len(features))
    if row_count < 10:
        raise HTTPException(status_code=400, detail="Dataset terlalu kecil. Minimal 10 baris.")
    if df[target].nunique() < 2:
        raise HTTPException(status_code=400, detail="Target harus memiliki minimal 2 kelas.")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset_name = f"dataset_{uuid.uuid4().hex}.csv"
    stored_path = DATA_DIR / dataset_name
    await asyncio.to_thread(df.to_csv, str(stored_path), index=False)
    dataset_id = db_utils.insert_dataset(
        conn,
        filename=src_path.name,
        stored_path=str(stored_path),
        row_count=row_count,
        feature_count=feature_count,
        target_column=target,
        uploaded_by=session_user.username,
    )
    db_utils.insert_activity(
        conn,
        actor_user_id=session_user.id,
        actor_username=session_user.username,
        actor_role=session_user.role,
        action="IMPORT_DATASET_LOCAL",
        detail={
            "dataset_id": dataset_id,
            "source_path": str(src_path),
            "stored_path": str(stored_path),
            "row_count": row_count,
            "feature_count": feature_count,
            "target": target,
        },
    )
    return JSONResponse(
        {
            "dataset_id": dataset_id,
            "filename": src_path.name,
            "stored_path": str(stored_path),
            "row_count": row_count,
            "feature_count": feature_count,
            "target": target,
        }
    )


def _pad_to_min_lines() -> None:
    return


_pad_to_min_lines()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
