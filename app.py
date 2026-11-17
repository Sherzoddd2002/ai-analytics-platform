"""
AI Analytics Platform - optimized version.
Key fixes: cached profiling/charts/insights, sampled analytics, gzip,
bounded sessions, safer error handling.
"""
import os

# Joblib/scikit-learn should not spawn `wmic` on Windows just to count cores.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import gzip
import json
import uuid
import time
import threading
import traceback
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask, render_template, request, jsonify, send_file
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score,
    classification_report, silhouette_score,
)
from sklearn.decomposition import PCA
import scipy.stats as stats
from werkzeug.utils import secure_filename

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ---- Bounds that keep the UI responsive ----
MAX_STORED_ROWS = 250_000
MAX_STORED_COLS = 200
PROFILE_SAMPLE_ROWS = 50_000
CHART_SAMPLE_ROWS = 2_000
CUSTOM_CHART_SAMPLE_ROWS = 5_000
ML_SAMPLE_ROWS = 10_000
CV_SAMPLE_ROWS = 3_000
CLUSTER_SAMPLE_ROWS = 10_000
INSIGHT_SAMPLE_ROWS = 10_000
SESSION_TTL_HOURS = 6
MAX_SESSIONS = 20

SESSIONS = {}


# ========================= HELPERS =========================

def _now():
    return datetime.now()


def _remove_file(path):
    try:
        if path and os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass


def _cleanup_sessions():
    now = _now()
    stale = [
        sid for sid, s in SESSIONS.items()
        if now - s["last_access"] > timedelta(hours=SESSION_TTL_HOURS)
    ]
    if len(SESSIONS) - len(stale) > MAX_SESSIONS:
        ordered = sorted(SESSIONS.items(), key=lambda kv: kv[1]["last_access"])
        extra = [sid for sid, s in ordered if sid not in stale]
        stale += extra[:len(SESSIONS) - len(stale) - MAX_SESSIONS]
    for sid in stale:
        s = SESSIONS.pop(sid, None)
        if s:
            _remove_file(s.get("filepath"))


def _get_session(sid):
    s = SESSIONS.get(sid)
    if s:
        s["last_access"] = _now()
    return s


def _body():
    return request.get_json(silent=True) or {}


def _fail(e, status=500):
    print(f"[ERROR] {type(e).__name__}: {e}\n{traceback.format_exc()}", flush=True)
    return jsonify({"error": str(e)}), status


def _sample(df, limit):
    if limit and len(df) > limit:
        return df.sample(n=limit, random_state=42).sort_index()
    return df


def _fig_json(fig):
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def _style(fig, title, height=300):
    fig.update_layout(
        title=title,
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0", size=11),
        margin=dict(l=30, r=15, t=40, b=30),
        height=height,
    )
    return fig


@app.before_request
def _before():
    _cleanup_sessions()


@app.after_request
def _gzip_json(response):
    """Plotly JSON compresses ~10x: big win on upload/dashboard."""
    if (
        response.status_code == 200
        and response.mimetype == "application/json"
        and "gzip" in request.headers.get("Accept-Encoding", "").lower()
        and not response.headers.get("Content-Encoding")
    ):
        raw = response.get_data()
        if len(raw) > 200 * 1024:
            response.set_data(gzip.compress(raw, compresslevel=5))
            response.headers["Content-Encoding"] = "gzip"
            response.headers["Vary"] = "Accept-Encoding"
    return response


# ========================= ANALYTICS CORE =========================

def generate_data_profiling(df):
    src = _sample(df, PROFILE_SAMPLE_ROWS)
    profiling = {
        "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
        "columns": {},
        "quality": {},
    }
    null_counts = df.isnull().sum()
    for col in df.columns:
        s = src[col]
        ci = {
            "dtype": str(df[col].dtype),
            "non_null": int(df[col].count()),
            "null_count": int(null_counts[col]),
            "null_pct": float(null_counts[col] / max(len(df), 1) * 100),
            "unique_count": int(s.nunique()),
            "is_numeric": bool(pd.api.types.is_numeric_dtype(df[col])),
            "is_categorical": bool(df[col].dtype == "object" or str(df[col].dtype) == "category"),
            "is_datetime": bool(pd.api.types.is_datetime64_any_dtype(df[col])),
        }
        if ci["is_numeric"]:
            for key, fn in (
                ("min", "min"), ("max", "max"), ("mean", "mean"),
                ("median", "median"), ("std", "std"), ("skew", "skew"),
            ):
                v = getattr(s, fn)()
                ci[key] = float(v) if pd.notna(v) else None
        if ci["is_categorical"]:
            tv = s.value_counts().head(5)
            ci["top_values"] = {str(k): int(v) for k, v in tv.items()}
        profiling["columns"][col] = ci

    complete_rows = int(df.notna().all(axis=1).sum())
    profiling["quality"] = {
        "total_missing": int(null_counts.sum()),
        "missing_pct": float(null_counts.mean().mean() * 100),
        "duplicates": int(df.duplicated().sum()),
        "complete_rows": complete_rows,
        "complete_pct": float(complete_rows / max(len(df), 1) * 100),
    }
    return profiling


def generate_preview(df, page=1, per_page=50):
    per_page = max(1, min(int(per_page or 50), 200))
    page = max(1, int(page or 1))
    total = len(df)
    start = (page - 1) * per_page
    end = min(start + per_page, total)
    df_page = df.iloc[start:end]
    records = []
    for _, row in df_page.iterrows():
        rec = {}
        for col in df.columns:
            val = row[col]
            if pd.isna(val):
                rec[col] = None
            elif isinstance(val, np.integer):
                rec[col] = int(val)
            elif isinstance(val, np.floating):
                rec[col] = float(val)
            elif isinstance(val, np.bool_):
                rec[col] = bool(val)
            else:
                rec[col] = str(val)
        records.append(rec)
    return {
        "data": records,
        "columns": [str(c) for c in df.columns],
        "page": page,
        "per_page": per_page,
        "total": total,
        "total_pages": (total + per_page - 1) // per_page,
    }


def generate_auto_charts(df):
    """Every trace is sampled: no 100k-point JSON blobs."""
    src = _sample(df, CHART_SAMPLE_ROWS)
    charts = []
    numeric_cols = src.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = src.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = [c for c in src.columns if pd.api.types.is_datetime64_any_dtype(src[c])]

    for col in src.columns:
        if col in datetime_cols or src[col].dtype != "object":
            continue
        try:
            pd.to_datetime(src[col].dropna().head(50), errors="raise")
            datetime_cols.append(col)
        except Exception:
            pass

    for col in numeric_cols[:4]:
        fig = _style(px.histogram(src, x=col, nbins=30,
                                  color_discrete_sequence=["#6366f1"]),
                     f"Taqsimot: {col}")
        charts.append({"title": f"Taqsimot: {col}", "type": "histogram",
                       "chart": _fig_json(fig)})

    for col in cat_cols[:3]:
        if src[col].nunique() <= 30:
            tv = src[col].value_counts().head(15)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=tv.index.astype(str), y=tv.values,
                                 marker=dict(color="#10b981")))
            _style(fig, f"Frekans: {col}")
            fig.update_xaxes(tickangle=-30)
            charts.append({"title": f"Frekans: {col}", "type": "bar",
                           "chart": _fig_json(fig)})

    corr_cols = numeric_cols[:20]
    if len(corr_cols) >= 2:
        corr = src[corr_cols].corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale="RdBu", zmid=0,
            text=corr.values.round(2), texttemplate="%{text}",
            hovertemplate="%{x} vs %{y}: %{z:.3f}<extra></extra>"))
        _style(fig, "Korrelyatsiya matrisasi", 350)
        charts.append({"title": "Korrelyatsiya matrisasi", "type": "heatmap",
                       "chart": _fig_json(fig)})

    if len(numeric_cols) >= 2:
        fig = _style(px.scatter(src, x=numeric_cols[0], y=numeric_cols[1],
                                color_discrete_sequence=["#f59e0b"], opacity=0.6),
                     f"Scatter: {numeric_cols[1]} vs {numeric_cols[0]}")
        charts.append({"title": f"Scatter: {numeric_cols[1]} vs {numeric_cols[0]}",
                       "type": "scatter", "chart": _fig_json(fig)})

    if datetime_cols and numeric_cols:
        dt_col, num_col = datetime_cols[0], numeric_cols[0]
        try:
            tmp = src.copy()
            tmp[dt_col] = pd.to_datetime(tmp[dt_col], errors="coerce")
            tmp = tmp.dropna(subset=[dt_col, num_col]).sort_values(dt_col)
            if len(tmp):
                fig = _style(px.line(tmp, x=dt_col, y=num_col,
                                     color_discrete_sequence=["#ec4899"]),
                             f"Vaqt qatori: {num_col} / {dt_col}")
                charts.append({"title": f"Vaqt qatori: {num_col} / {dt_col}",
                               "type": "line", "chart": _fig_json(fig)})
        except Exception:
            pass

    if numeric_cols:
        fig = go.Figure()
        for c in numeric_cols[:6]:
            fig.add_trace(go.Box(y=src[c].dropna(), name=c))
        _style(fig, "Box: sonli ustunlar", 320)
        charts.append({"title": "Box: sonli ustunlar", "type": "box",
                       "chart": _fig_json(fig)})

    return charts


def get_columns_info(df):
    src = _sample(df, PROFILE_SAMPLE_ROWS)
    info = []
    for col in df.columns:
        s = src[col]
        info.append({
            "name": str(col), "dtype": str(df[col].dtype),
            "n_unique": int(s.nunique()),
            "n_missing": int(df[col].isnull().sum()),
            "is_numeric": bool(pd.api.types.is_numeric_dtype(df[col])),
            "is_categorical": bool(df[col].dtype == "object" or str(df[col].dtype) == "category"),
            "sample_values": [str(v)[:80] for v in s.dropna().head(3).tolist()],
        })
    return info


def cached(sess, key, builder):
    if key not in sess["cache"]:
        sess["cache"][key] = builder()
    return sess["cache"][key]


def invalidate(sess):
    sess["cache"] = {}


def read_table(filepath, ext):
    if ext in (".csv", ".tsv"):
        sep = "\t" if ext == ".tsv" else ","
        try:
            return pd.read_csv(filepath, sep=sep, encoding="utf-8-sig", low_memory=False)
        except UnicodeDecodeError:
            return pd.read_csv(filepath, sep=sep, encoding="cp1252", low_memory=False)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(filepath)
    if ext == ".json":
        return pd.read_json(filepath)
    raise ValueError(f"Format qo'llab-quvvatlanmaydi: {ext}")


# ========================= ROUTES =========================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        # A very early upload may arrive while the background warmup runs.
        _WARMUP_DONE.wait(timeout=20)
        if "file" not in request.files:
            return jsonify({"error": "Fayl topilmadi"}), 400
        file = request.files["file"]
        if not file.filename:
            return jsonify({"error": "Fayl tanlanmagan"}), 400

        filename = secure_filename(os.path.basename(file.filename))
        ext = os.path.splitext(filename)[1].lower()
        if ext not in (".csv", ".xlsx", ".xls", ".json", ".tsv"):
            return jsonify({"error": "Faqat CSV, Excel, JSON yoki TSV"}), 400

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], f"{uuid.uuid4().hex}_{filename}")
        file.save(filepath)
        try:
            df = read_table(filepath, ext)
        except Exception:
            _remove_file(filepath)
            raise

        if df.empty:
            _remove_file(filepath)
            return jsonify({"error": "Fayl bo'sh yoki ma'lumot yo'q"}), 400

        note = None
        if len(df) > MAX_STORED_ROWS or len(df.columns) > MAX_STORED_COLS:
            note = (f"Fayl qisqartirildi: {len(df):,} qator / {len(df.columns)} ustundan "
                    f"{min(len(df), MAX_STORED_ROWS):,} / {min(len(df.columns), MAX_STORED_COLS)} gacha.")
            df = df.iloc[:MAX_STORED_ROWS, :MAX_STORED_COLS].copy()

        sid = uuid.uuid4().hex
        SESSIONS[sid] = {
            "df": df,
            "original_df": df.copy(),
            "filename": filename,
            "filepath": filepath,
            "created_at": _now().isoformat(),
            "last_access": _now(),
            "history": [],
            "cache": {},
            "note": note,
        }
        sess = SESSIONS[sid]
        profiling = cached(sess, "profiling", lambda: generate_data_profiling(df))
        preview = cached(sess, "preview", lambda: generate_preview(df))
        charts = cached(sess, "charts", lambda: generate_auto_charts(df))
        columns = cached(sess, "columns", lambda: get_columns_info(df))

        return jsonify({
            "session_id": sid,
            "filename": filename,
            "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
            "columns": [c["name"] for c in columns],
            "note": note,
            "profiling": profiling,
            "preview": preview,
            "auto_charts": charts,
        })
    except Exception as e:
        return _fail(e)


@app.route("/profile", methods=["POST"])
def get_profiling():
    try:
        sess = _get_session(_body().get("session_id"))
        if not sess:
            return jsonify({"error": "Sessiya topilmadi", "session_expired": True}), 400
        return jsonify({"profiling": cached(
            sess, "profiling", lambda: generate_data_profiling(sess["df"]))})
    except Exception as e:
        return _fail(e)


@app.route("/preview", methods=["POST"])
def get_preview():
    try:
        d = _body()
        sess = _get_session(d.get("session_id"))
        if not sess:
            return jsonify({"error": "Sessiya topilmadi", "session_expired": True}), 400
        key = f"preview:{d.get('page', 1)}:{d.get('per_page', 50)}"
        return jsonify({"preview": cached(
            sess, key,
            lambda: generate_preview(sess["df"], d.get("page", 1), d.get("per_page", 50)))})
    except Exception as e:
        return _fail(e)


@app.route("/columns", methods=["POST"])
def get_columns():
    try:
        sess = _get_session(_body().get("session_id"))
        if not sess:
            return jsonify({"error": "Sessiya topilmadi", "session_expired": True}), 400
        return jsonify({"columns": cached(
            sess, "columns", lambda: get_columns_info(sess["df"]))})
    except Exception as e:
        return _fail(e)


@app.route("/charts/auto", methods=["POST"])
def auto_charts():
    try:
        sess = _get_session(_body().get("session_id"))
        if not sess:
            return jsonify({"error": "Sessiya topilmadi", "session_expired": True}), 400
        return jsonify({"auto_charts": cached(
            sess, "charts", lambda: generate_auto_charts(sess["df"]))})
    except Exception as e:
        return _fail(e)


@app.route("/clean", methods=["POST"])
def clean_data():
    try:
        d = _body()
        sess = _get_session(d.get("session_id"))
        if not sess:
            return jsonify({"error": "Sessiya topilmadi", "session_expired": True}), 400
        df = sess["df"].copy()
        operations = d.get("operations", [])
        results = []
        for op in operations:
            t = op.get("type")
            if t == "drop_duplicates":
                before = len(df)
                df = df.drop_duplicates()
                results.append(f"Dublikatlar o'chirildi: {before - len(df)} ta qator")
            elif t == "drop_missing":
                thr = float(op.get("threshold", 0.5))
                bc = df.shape[1]
                df = df.loc[:, df.isnull().mean() <= thr]
                dc = bc - df.shape[1]
                br = len(df)
                df = df.dropna()
                results.append(f"Yetishmayotgan tozalandi: {dc} ustun, {br - len(df)} qator")
            elif t == "fill_missing":
                strat = op.get("strategy", "mean")
                filled = 0
                for col in df.columns:
                    if df[col].isnull().any():
                        if pd.api.types.is_numeric_dtype(df[col]):
                            v = df[col].mean() if strat == "mean" else df[col].median() if strat == "median" else 0
                        else:
                            modes = df[col].mode()
                            v = modes.iloc[0] if len(modes) else "Unknown"
                        filled += int(df[col].isnull().sum())
                        df[col] = df[col].fillna(v)
                results.append(f"Yetishmayotgan to'ldirildi: {filled} ta ({strat})")
            elif t == "remove_outliers":
                m = op.get("method", "iqr")
                removed = 0
                for col in df.select_dtypes(include=[np.number]).columns:
                    if m == "iqr":
                        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                        iqr = q3 - q1
                        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                        mask = (df[col] >= lo) & (df[col] <= hi)
                        removed += int((~mask).sum())
                        df = df[mask]
                    elif m == "zscore":
                        sub = df[col].dropna()
                        z = np.abs(stats.zscore(sub))
                        keep_idx = sub.index[z < float(op.get("threshold", 3))]
                        df = df.loc[df.index.isin(keep_idx) | df[col].isnull()]
                results.append(f"Anomal qiymatlar: {removed} ta ({m})")
            elif t == "normalize":
                m = op.get("method", "minmax")
                for col in df.select_dtypes(include=[np.number]).columns:
                    if m == "minmax":
                        lo, hi = df[col].min(), df[col].max()
                        if hi - lo > 0:
                            df[col] = (df[col] - lo) / (hi - lo)
                    elif m == "standard":
                        mu, sd = df[col].mean(), df[col].std()
                        if sd > 0:
                            df[col] = (df[col] - mu) / sd
                results.append(f"Normalizatsiya: {m}")
            elif t == "encode_categorical":
                m = op.get("method", "label")
                for col in list(df.select_dtypes(include=["object", "category"]).columns):
                    if m == "label":
                        df[col] = pd.factorize(df[col].astype(str))[0]
                    elif m == "onehot":
                        dummies = pd.get_dummies(df[col], prefix=str(col), drop_first=True)
                        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                results.append(f"Kategoriyali kodlandi: {m}")
            elif t == "drop_column":
                c = op.get("column")
                if c in df.columns:
                    df = df.drop(columns=[c])
                    results.append(f"Ustun o'chirildi: {c}")
            elif t == "rename_column":
                o, n = op.get("old_name"), op.get("new_name")
                if o in df.columns:
                    df = df.rename(columns={o: n})
                    results.append(f"Nom o'zgartirildi: {o} -> {n}")
            elif t == "change_type":
                c, nt = op.get("column"), op.get("new_type")
                if c in df.columns:
                    try:
                        if nt == "int":
                            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
                        elif nt == "float":
                            df[c] = pd.to_numeric(df[c], errors="coerce")
                        elif nt == "str":
                            df[c] = df[c].astype(str)
                        elif nt == "datetime":
                            df[c] = pd.to_datetime(df[c], errors="coerce")
                        results.append(f"Tur o'zgartirildi: {c} -> {nt}")
                    except Exception as ex:
                        results.append(f"Xato ({c}): {ex}")

        sess["df"] = df
        sess["history"].append({
            "action": "clean", "operations": operations,
            "results": results, "timestamp": _now().isoformat(),
        })
        invalidate(sess)
        # Charts are fetched separately from /charts/auto.
        return jsonify({
            "success": True,
            "results": results,
            "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
            "profiling": cached(sess, "profiling", lambda: generate_data_profiling(df)),
            "preview": cached(sess, "preview", lambda: generate_preview(df)),
        })
    except Exception as e:
        return _fail(e)


@app.route("/chart", methods=["POST"])
def create_chart():
    try:
        d = _body()
        sess = _get_session(d.get("session_id"))
        if not sess:
            return jsonify({"error": "Sessiya topilmadi", "session_expired": True}), 400
        src = _sample(sess["df"], CUSTOM_CHART_SAMPLE_ROWS)
        ct = d.get("chart_type")
        # The UI sends "" for optional selects; Plotly would treat it as a
        # column name and fail. Normalize empty strings to None.
        x = d.get("x_col") or None
        y = d.get("y_col") or None
        z = d.get("z_col") or None
        cc = d.get("color_col") or None
        sc = d.get("size_col") or None
        af = d.get("agg_func") or "sum"
        title = d.get("title") or ""
        fig = None
        if ct == "scatter":
            fig = px.scatter(src, x=x, y=y, color=cc, size=sc, title=title or f"{y} vs {x}")
        elif ct == "line":
            fig = px.line(src, x=x, y=y, color=cc, title=title or f"{y} over {x}")
        elif ct == "bar":
            if af != "none" and y:
                g = src.groupby(x)[y].agg(af).reset_index()
                fig = px.bar(g, x=x, y=y, color=cc, title=title or f"{y} by {x} ({af})")
            else:
                fig = px.bar(src, x=x, y=y, color=cc, title=title or f"Bar: {x}")
        elif ct == "histogram":
            fig = px.histogram(src, x=x, color=cc, title=title or f"Distribution: {x}")
        elif ct == "box":
            fig = px.box(src, x=x, y=y, color=cc, title=title or f"Box: {y} by {x}")
        elif ct == "violin":
            fig = px.violin(src, x=x, y=y, color=cc, box=True, title=title or f"Violin: {y} by {x}")
        elif ct == "pie":
            if y and af != "none":
                g = src.groupby(x)[y].agg(af).reset_index()
            else:
                g = src[x].value_counts().reset_index()
                g.columns = [x, "count"]
            fig = px.pie(g, names=x, values=g.columns[-1], title=title or f"Pie: {x}")
        elif ct == "heatmap":
            ndf = src.select_dtypes(include=[np.number]).iloc[:, :20]
            if len(ndf.columns):
                corr = ndf.corr()
                fig = go.Figure(data=go.Heatmap(
                    z=corr.values, x=corr.columns, y=corr.columns,
                    colorscale="RdBu", zmid=0,
                    text=corr.values.round(3), texttemplate="%{text}",
                    hovertemplate="%{x} vs %{y}: %{z:.3f}<extra></extra>"))
                fig.update_layout(title=title or "Korrelyatsiya")
        elif ct == "scatter3d":
            fig = px.scatter_3d(src, x=x, y=y, z=z, color=cc, title=title or "3D")
        elif ct == "area":
            fig = px.area(src, x=x, y=y, color=cc, title=title or "Area")
        elif ct == "sunburst":
            path = [x, cc] if cc else [x]
            fig = px.sunburst(src, path=path, values=y if y else None, title=title or "Sunburst")
        elif ct == "treemap":
            path = [x, cc] if cc else [x]
            fig = px.treemap(src, path=path, values=y if y else None, title=title or "Treemap")
        else:
            return jsonify({"error": f"Noma'lum grafik: {ct}"}), 400

        if fig:
            fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
                              paper_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="#e0e0e0", size=12),
                              margin=dict(l=40, r=20, t=50, b=40))
            return jsonify({"chart": _fig_json(fig)})
        return jsonify({"error": "Grafik yaratilmadi"}), 400
    except Exception as e:
        return _fail(e)


def _prepare_xy(sess, d):
    df = _sample(sess["df"], ML_SAMPLE_ROWS)
    target = d.get("target")
    if target not in df.columns:
        raise ValueError(f"Maqsad ustun topilmadi: {target}")
    ptype = d.get("problem_type", "auto")
    if ptype == "auto":
        ratio = df[target].nunique() / max(len(df), 1)
        ptype = "classification" if (
            pd.api.types.is_object_dtype(df[target]) or
            pd.api.types.is_categorical_dtype(df[target]) or
            (pd.api.types.is_integer_dtype(df[target]) and ratio < 0.05)
        ) else "regression"
    features = d.get("features") or [c for c in df.columns if c != target]
    X = df[features].copy()
    y = df[target].copy()

    for col in X.columns:
        if pd.api.types.is_object_dtype(X[col]) or str(X[col].dtype) == "category":
            s = X[col].astype(str)
            if s.nunique() > 50:
                X[col] = s.map(s.value_counts().to_dict())
            else:
                X[col] = pd.factorize(s)[0]
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True)).fillna(0)

    if ptype == "classification":
        codes, uniques = pd.factorize(y.astype(str))
        y = codes
        class_names = [str(c) for c in uniques]
    else:
        y = pd.to_numeric(y, errors="coerce")
        ok = y.notna()
        X, y = X.loc[ok], y.loc[ok]
        class_names = []
    return X, y, ptype, class_names


def _cv_score(model, X, y, scoring):
    Xs = _sample(X, CV_SAMPLE_ROWS)
    ys = y.loc[Xs.index]
    if len(Xs) < 20 or ys.nunique() < 2:
        return 0.0, 0.0
    try:
        cv = cross_val_score(model, Xs, ys, cv=3, scoring=scoring)
        return float(np.nanmean(cv)), float(np.nanstd(cv))
    except Exception:
        return 0.0, 0.0


@app.route("/ml/predict", methods=["POST"])
def ml_predict():
    try:
        d = _body()
        sess = _get_session(d.get("session_id"))
        if not sess:
            return jsonify({"error": "Sessiya topilmadi", "session_expired": True}), 400
        X, y, ptype, class_names = _prepare_xy(sess, d)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=float(d.get("test_size", 0.2)), random_state=42)

        model_type = d.get("model_type", "auto")
        if ptype == "regression":
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(n_estimators=40, random_state=42, n_jobs=-1),
                "Gradient Boosting": GradientBoostingRegressor(n_estimators=40, random_state=42),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
            }
            mapping = {"linear": "Linear Regression", "random_forest": "Random Forest",
                       "gradient_boosting": "Gradient Boosting", "decision_tree": "Decision Tree"}
            scoring = "r2"
        else:
            models = {
                "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
                "Random Forest": RandomForestClassifier(n_estimators=40, random_state=42, n_jobs=-1),
                "Gradient Boosting": GradientBoostingClassifier(n_estimators=40, random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
            }
            mapping = {"logistic": "Logistic Regression", "random_forest": "Random Forest",
                       "gradient_boosting": "Gradient Boosting", "decision_tree": "Decision Tree"}
            scoring = "accuracy"

        if model_type != "auto":
            name = mapping.get(model_type, "Random Forest")
            models = {name: models[name]}

        best_model, best_score, best_name, all_res = None, -np.inf, "", {}
        for name, mdl in models.items():
            t0 = time.perf_counter()
            try:
                mdl.fit(X_train, y_train)
                sc = mdl.score(X_test, y_test)
                all_res[name] = {
                    ("r2" if ptype == "regression" else "accuracy"): float(sc),
                    "seconds": round(time.perf_counter() - t0, 2),
                }
                if sc > best_score:
                    best_score, best_model, best_name = sc, mdl, name
            except Exception as ex:
                all_res[name] = {"error": str(ex)[:120]}
        if best_model is None:
            return jsonify({"error": "Hech bir model o'rgatilmadi", "models": all_res}), 400

        # No second full fit: best_model is already fitted.
        y_pred = best_model.predict(X_test)
        cv_mean, cv_std = _cv_score(best_model, X, y, scoring)

        if hasattr(best_model, "feature_importances_"):
            imps = best_model.feature_importances_
        elif hasattr(best_model, "coef_"):
            c = best_model.coef_
            imps = np.abs(c[0]) if getattr(c, "ndim", 1) > 1 else np.abs(c)
        else:
            imps = np.zeros(X.shape[1])
        fi = {str(col): float(imps[i]) if i < len(imps) else 0.0
              for i, col in enumerate(X.columns)}
        fi = dict(sorted(fi.items(), key=lambda kv: kv[1], reverse=True))

        imp_fig = go.Figure()
        fi_cols = list(fi)[:15]
        imp_fig.add_trace(go.Bar(x=[fi[c] for c in fi_cols], y=fi_cols,
                                 orientation="h", marker=dict(color="#6366f1")))
        imp_fig.update_layout(title="Feature Importance", template="plotly_dark",
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="#e0e0e0"), margin=dict(l=120))
        imp_fig.update_yaxes(autorange="reversed")
        response = {
            "success": True, "problem_type": ptype, "model_name": best_name,
            "class_names": class_names, "all_models": all_res,
            "feature_importance": fi, "importance_chart": _fig_json(imp_fig),
            "n_features": int(X.shape[1]), "n_train": int(len(X_train)),
            "n_test": int(len(X_test)), "n_used": int(len(X)),
        }

        if ptype == "regression":
            response["metrics"] = {
                "r2": float(r2_score(y_test, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "cv_mean": cv_mean, "cv_std": cv_std,
            }
            pred_fig = go.Figure()
            pred_fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers",
                                          name="Bashorat vs Haqiqiy",
                                          marker=dict(color="#6366f1", size=6, opacity=0.6)))
            lo = float(min(y_test.min(), y_pred.min()))
            hi = float(max(y_test.max(), y_pred.max()))
            pred_fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                          name="Ideal", line=dict(color="#ef4444", dash="dash")))
            pred_fig.update_layout(title="Bashorat vs Haqiqiy", xaxis_title="Haqiqiy",
                                   yaxis_title="Bashorat", template="plotly_dark",
                                   plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                   font=dict(color="#e0e0e0"))
            response["prediction_chart"] = _fig_json(pred_fig)
        else:
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            response["metrics"] = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "cv_mean": cv_mean, "cv_std": cv_std,
            }
            response["classification_report"] = report
            classes = [k for k in report if k not in ("accuracy", "macro avg", "weighted avg")]
            rep_fig = go.Figure()
            for metric in ("precision", "recall", "f1-score"):
                rep_fig.add_trace(go.Bar(name=metric, x=classes,
                                         y=[report[c].get(metric, 0) for c in classes]))
            rep_fig.update_layout(barmode="group", title="Klassifikatsiya hisoboti",
                                  template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
                                  paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e0e0e0"))
            response["report_chart"] = _fig_json(rep_fig)
        return jsonify(response)
    except Exception as e:
        return _fail(e)


@app.route("/ml/cluster", methods=["POST"])
def ml_cluster():
    try:
        d = _body()
        sess = _get_session(d.get("session_id"))
        if not sess:
            return jsonify({"error": "Sessiya topilmadi", "session_expired": True}), 400
        k = int(d.get("n_clusters", 3))
        features = d.get("features") or []
        src = _sample(sess["df"], CLUSTER_SAMPLE_ROWS)
        X = src[features].copy() if features else src.select_dtypes(include=[np.number]).copy()
        if X.empty:
            return jsonify({"error": "Klasterlash uchun sonli ustun kerak"}), 400
        for col in X.columns:
            if pd.api.types.is_object_dtype(X[col]) or str(X[col].dtype) == "category":
                X[col] = pd.factorize(X[col].astype(str))[0]
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.fillna(X.median(numeric_only=True)).fillna(0)
        X_s = StandardScaler().fit_transform(X)

        is_big = len(X_s) > 5_000
        cls = (MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1024, n_init=3)
               if is_big else KMeans(n_clusters=k, random_state=42, n_init=5))
        clusters = cls.fit_predict(X_s)

        sil = 0.0
        if k > 1 and len(X_s) > 2:
            idx = np.linspace(0, len(X_s) - 1, min(2000, len(X_s)), dtype=int)
            try:
                sil = float(silhouette_score(X_s[idx], clusters[idx]))
            except Exception:
                sil = 0.0

        pca = PCA(n_components=min(3, X_s.shape[1]))
        X_pca = pca.fit_transform(X_s)
        if X_pca.shape[1] >= 3:
            fig = px.scatter_3d(x=X_pca[:, 0], y=X_pca[:, 1], z=X_pca[:, 2],
                                color=clusters.astype(str), title=f"Klasterlash (K={k})",
                                color_discrete_sequence=px.colors.qualitative.Set2)
        else:
            fig = px.scatter(x=X_pca[:, 0],
                             y=X_pca[:, 1] if X_pca.shape[1] > 1 else [0] * len(X_pca),
                             color=clusters.astype(str), title=f"Klasterlash (K={k})",
                             color_discrete_sequence=px.colors.qualitative.Set2)
        _style(fig, f"Klasterlash (K={k})")
        chart = _fig_json(fig)

        k_max = min(8, max(2, len(X_s) - 1))
        distortions, K_range = [], list(range(1, k_max + 1))
        for kk in K_range:
            km = (MiniBatchKMeans(n_clusters=kk, random_state=42, batch_size=1024, n_init=2)
                  if is_big else KMeans(n_clusters=kk, random_state=42, n_init=3))
            km.fit(X_s)
            distortions.append(km.inertia_)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=K_range, y=distortions, mode="lines+markers",
                                  marker=dict(color="#6366f1", size=8)))
        _style(fig2, "Elbow Method")
        fig2.update_layout(xaxis_title="K", yaxis_title="Inertia")

        summary = {}
        tmp = src.copy()
        tmp["_cluster"] = clusters
        for c in range(k):
            cd = tmp[tmp["_cluster"] == c]
            entry = {"size": int(len(cd)),
                     "percentage": float(len(cd) / max(len(tmp), 1) * 100)}
            for col in X.columns:
                if pd.api.types.is_numeric_dtype(tmp[col]):
                    v = cd[col].mean()
                    entry[f"{col}_mean"] = float(v) if pd.notna(v) else 0.0
            summary[f"Klaster {c}"] = entry

        return jsonify({
            "success": True, "n_clusters": k, "silhouette_score": sil,
            "chart": chart, "elbow_chart": _fig_json(fig2),
            "cluster_summary": summary, "n_points": int(len(X_s)),
            "explained_variance": [float(v) for v in pca.explained_variance_ratio_],
        })
    except Exception as e:
        return _fail(e)


@app.route("/stats", methods=["POST"])
def statistics():
    try:
        d = _body()
        sess = _get_session(d.get("session_id"))
        if not sess:
            return jsonify({"error": "Sessiya topilmadi", "session_expired": True}), 400
        df = sess["df"]
        src = _sample(df, CUSTOM_CHART_SAMPLE_ROWS)
        at = d.get("analysis_type")
        c1, c2 = d.get("col1"), d.get("col2")
        results = {}
        if at == "correlation":
            ndf = src.select_dtypes(include=[np.number]).iloc[:, :20]
            if len(ndf.columns) < 2:
                return jsonify({"error": "Kamida 2 sonli ustun kerak"}), 400
            corr = ndf.corr()
            fig = go.Figure(data=go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.columns,
                colorscale="RdBu", zmid=0, text=corr.values.round(3),
                texttemplate="%{text}",
                hovertemplate="%{x} vs %{y}: %{z:.3f}<extra></extra>"))
            _style(fig, "Korrelyatsiya")
            results["chart"] = _fig_json(fig)
            results["correlation_matrix"] = {
                str(k): {str(k2): float(v2) for k2, v2 in v.items()}
                for k, v in corr.to_dict().items()
            }
        elif at == "ttest":
            if c1 and c2 and c1 in df.columns and c2 in df.columns:
                st, pv = stats.ttest_ind(df[c1].dropna(), df[c2].dropna())
                results.update({"t_statistic": float(st), "p_value": float(pv),
                                "significant": bool(pv < 0.05)})
        elif at == "distribution" and c1 and c1 in df.columns:
            cd = src[c1].dropna()
            if pd.api.types.is_numeric_dtype(cd):
                if len(cd) >= 3:
                    st, pv = stats.shapiro(cd.head(5000))
                    results.update({"shapiro_stat": float(st), "shapiro_p": float(pv),
                                    "normal": bool(pv > 0.05)})
                results["skewness"] = float(stats.skew(cd))
                results["kurtosis"] = float(stats.kurtosis(cd))
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=cd, name="Haqiqiy", opacity=0.7,
                                           marker_color="#6366f1"))
                mu, sd = cd.mean(), cd.std()
                if sd > 0 and len(cd):
                    xr = np.linspace(float(cd.min()), float(cd.max()), 100)
                    pdf = stats.norm.pdf(xr, mu, sd) * len(cd) * (float(cd.max()) - float(cd.min())) / 30
                    fig.add_trace(go.Scatter(x=xr, y=pdf, name="Normal",
                                             line=dict(color="#ef4444", width=2)))
                _style(fig, f"Taqsimot: {c1}")
                results["chart"] = _fig_json(fig)
        return jsonify({"success": True, "results": results})
    except Exception as e:
        return _fail(e)


@app.route("/ai_insights", methods=["POST"])
def ai_insights():
    try:
        d = _body()
        sess = _get_session(d.get("session_id"))
        if not sess:
            return jsonify({"error": "Sessiya topilmadi", "session_expired": True}), 400
        if "insights" not in sess["cache"]:
            sess["cache"]["insights"] = _build_insights(sess["df"])
        return jsonify({"insights": sess["cache"]["insights"]})
    except Exception as e:
        return _fail(e)


def _build_insights(df):
    src = _sample(df, INSIGHT_SAMPLE_ROWS)
    insights = []
    nc = src.select_dtypes(include=[np.number]).columns.tolist()
    cc = src.select_dtypes(include=["object", "category"]).columns.tolist()
    insights.append({
        "type": "info", "icon": "database", "title": "Ma'lumot hajmi",
        "text": f"Dataset {df.shape[0]:,} qator va {df.shape[1]} ustun. {len(nc)} sonli, {len(cc)} kategoriyali.",
    })
    missing = src.isnull().sum()
    if missing.sum() > 0:
        wc = missing.idxmax()
        wp = missing.max() / max(len(src), 1) * 100
        insights.append({
            "type": "warning", "icon": "alert-triangle",
            "title": "Yetishmayotgan ma'lumot",
            "text": f"Namunada {missing.sum():,} yetishmayotgan. Eng yomon: '{wc}' ({wp:.1f}%).",
        })
    else:
        insights.append({
            "type": "success", "icon": "check-circle",
            "title": "Ma'lumot to'liqligi", "text": "Yetishmayotgan qiymatlar yo'q.",
        })
    dup = src.duplicated().sum()
    if dup:
        insights.append({
            "type": "warning", "icon": "copy", "title": "Dublikatlar",
            "text": f"Namunada {dup} dublikat qator ({dup / max(len(src), 1) * 100:.1f}%).",
        })

    if len(nc) >= 2:
        corr_cols = nc[:20]
        corr = src[corr_cols].corr()
        pairs = []
        for i in range(len(corr_cols)):
            for j in range(i + 1, len(corr_cols)):
                pairs.append((corr_cols[i], corr_cols[j],
                              abs(corr.iloc[i, j]), corr.iloc[i, j]))
        pairs.sort(key=lambda p: p[2], reverse=True)
        if pairs and pairs[0][2] > 0.5:
            p = pairs[0]
            strength = "kuchli" if p[2] > 0.7 else "o'rtacha"
            direction = "musbat" if p[3] > 0 else "manfiy"
            insights.append({
                "type": "info", "icon": "trending-up", "title": "Korrelyatsiya",
                "text": f"{strength} {direction}: '{p[0]}' va '{p[1]}' (r={p[3]:.3f}).",
            })
    for col in cc:
        if src[col].nunique() > 50:
            insights.append({
                "type": "warning", "icon": "layers",
                "title": f"Yuqori kardinallik: {col}",
                "text": f"'{col}' ustunida {src[col].nunique()} noyob qiymat.",
            })
            break

    oc = 0
    for col in nc:
        q1, q3 = src[col].quantile(0.25), src[col].quantile(0.75)
        iqr = q3 - q1
        oc += int(((src[col] < q1 - 1.5 * iqr) | (src[col] > q3 + 1.5 * iqr)).sum())
    if oc:
        insights.append({
            "type": "warning", "icon": "zap", "title": "Anomal qiymatlar",
            "text": f"IQR bo'yicha ~{oc} anomal qiymat (namunada).",
        })
    for col in nc:
        sk = src[col].skew()
        if pd.notna(sk) and abs(sk) > 2:
            side = "o'ng" if sk > 0 else "chap"
            insights.append({
                "type": "info", "icon": "bar-chart",
                "title": f"Noto'g'ri taqsimot: {col}",
                "text": f"'{col}' {side} tomonga og'gan (skew={sk:.2f}).",
            })
            break

    if len(nc) >= 2:
        best_t, best_r = None, -np.inf
        for tgt in nc[:5]:
            feats = [c for c in nc[:8] if c != tgt]
            if not feats:
                continue
            td = src[feats + [tgt]].dropna().head(CV_SAMPLE_ROWS)
            if len(td) < 20:
                continue
            try:
                cv = cross_val_score(LinearRegression(), td[feats], td[tgt], cv=3, scoring="r2")
                m = float(np.nanmean(cv))
                if m > best_r:
                    best_r, best_t = m, tgt
            except Exception:
                continue
        if best_t and best_r > 0.3:
            insights.append({
                "type": "success", "icon": "target", "title": "ML tavsiyasi",
                "text": f"'{best_t}' bashorat qilish uchun yaxshi (R2~{best_r:.2f}).",
            })
    return insights


@app.route("/download", methods=["POST"])
def download_data():
    try:
        d = _body()
        sess = _get_session(d.get("session_id"))
        if not sess:
            return jsonify({"error": "Sessiya topilmadi", "session_expired": True}), 400
        fmt = d.get("format", "csv")
        if fmt not in ("csv", "xlsx", "json"):
            return jsonify({"error": "Noto'g'ri format"}), 400
        base = os.path.splitext(os.path.basename(sess["filename"]))[0] or "data"
        fp = os.path.join(
            app.config["UPLOAD_FOLDER"],
            f"{uuid.uuid4().hex}_{secure_filename(base)}_cleaned.{fmt}",
        )
        df = sess["df"]
        if fmt == "csv":
            df.to_csv(fp, index=False)
        elif fmt == "xlsx":
            df.to_excel(fp, index=False)
        else:
            df.to_json(fp, orient="records", indent=2)
        return send_file(fp, as_attachment=True, download_name=f"{base}_cleaned.{fmt}")
    except Exception as e:
        return _fail(e)


# ---- Warm up Plotly/scikit-learn in the background -------------------------
# The very first Plotly Express call costs ~7 s (template loading). If we let it
# happen on the first upload, the UI looks frozen. Do it once at startup instead.
_WARMUP_DONE = threading.Event()


def _warm_up():
    try:
        warm = pd.DataFrame({
            "warm_a": np.linspace(0, 1, 100),
            "warm_b": np.linspace(1, 0, 100),
            "warm_cat": ["x", "y"] * 50,
        })
        generate_data_profiling(warm)
        generate_preview(warm)
        generate_auto_charts(warm)
        LinearRegression().fit(np.arange(20).reshape(10, 2), np.arange(10))
    except Exception as exc:
        print(f"[WARN] warmup failed: {exc}", flush=True)
    finally:
        _WARMUP_DONE.set()


threading.Thread(target=_warm_up, name="warmup", daemon=True).start()


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))
    try:
        # Production WSGI server: threaded, no dev-server warning/reloader.
        from waitress import serve
        serve(app, host=host, port=port, threads=8)
    except ImportError:
        app.run(debug=False, threaded=True, host=host, port=port)
