"""
AI Analytics Platform - Julius.ai style
Flask backend with Pandas, Plotly, Scikit-learn
Author: Sherzod
"""
import os
import json
import uuid
import traceback
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask, render_template, request, jsonify, send_file
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, silhouette_score
from sklearn.decomposition import PCA
import scipy.stats as stats

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

SESSIONS = {}


def generate_data_profiling(df):
    profiling = {
        "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
        "columns": {},
        "quality": {}
    }
    for col in df.columns:
        ci = {
            "dtype": str(df[col].dtype),
            "non_null": int(df[col].count()),
            "null_count": int(df[col].isnull().sum()),
            "null_pct": float(df[col].isnull().mean() * 100),
            "unique_count": int(df[col].nunique()),
            "is_numeric": pd.api.types.is_numeric_dtype(df[col]),
            "is_categorical": df[col].dtype == "object" or str(df[col].dtype) == "category",
            "is_datetime": pd.api.types.is_datetime64_any_dtype(df[col]),
        }
        if ci["is_numeric"]:
            ci["min"] = float(df[col].min()) if not pd.isna(df[col].min()) else None
            ci["max"] = float(df[col].max()) if not pd.isna(df[col].max()) else None
            ci["mean"] = float(df[col].mean()) if not pd.isna(df[col].mean()) else None
            ci["median"] = float(df[col].median()) if not pd.isna(df[col].median()) else None
            ci["std"] = float(df[col].std()) if not pd.isna(df[col].std()) else None
            ci["skew"] = float(df[col].skew()) if not pd.isna(df[col].skew()) else None
        if ci["is_categorical"]:
            tv = df[col].value_counts().head(5)
            ci["top_values"] = {str(k): int(v) for k, v in tv.items()}
        profiling["columns"][col] = ci
    profiling["quality"]["total_missing"] = int(df.isnull().sum().sum())
    profiling["quality"]["missing_pct"] = float(df.isnull().mean().mean() * 100)
    profiling["quality"]["duplicates"] = int(df.duplicated().sum())
    profiling["quality"]["complete_rows"] = int(df.dropna().shape[0])
    profiling["quality"]["complete_pct"] = float(df.dropna().shape[0] / max(df.shape[0], 1) * 100)
    return profiling


def generate_preview(df, page=1, per_page=50):
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
            elif isinstance(val, (np.integer,)):
                rec[col] = int(val)
            elif isinstance(val, (np.floating,)):
                rec[col] = float(val)
            elif isinstance(val, (np.bool_,)):
                rec[col] = bool(val)
            else:
                rec[col] = str(val)
        records.append(rec)
    return {
        "data": records,
        "columns": list(df.columns),
        "page": page,
        "per_page": per_page,
        "total": total,
        "total_pages": (total + per_page - 1) // per_page,
    }


def generate_auto_charts(df):
    charts = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    for col in df.columns:
        if col in datetime_cols:
            continue
        if df[col].dtype == "object":
            try:
                pd.to_datetime(df[col].dropna().head(100), errors="raise")
                datetime_cols.append(col)
            except:
                pass

    def style_fig(fig, title, height=300):
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

    for col in numeric_cols[:4]:
        fig = px.histogram(df, x=col, nbins=30, color_discrete_sequence=["#6366f1"])
        fig = style_fig(fig, f"Taqsimot: {col}")
        charts.append({"title": f"Taqsimot: {col}", "type": "histogram",
                       "chart": json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)})

    for col in cat_cols[:3]:
        if df[col].nunique() <= 30:
            tv = df[col].value_counts().head(15)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=tv.index.astype(str), y=tv.values,
                                 marker=dict(color="#10b981")))
            fig = style_fig(fig, f"Frekans: {col}")
            fig.update_xaxes(tickangle=-30)
            charts.append({"title": f"Frekans: {col}", "type": "bar",
                           "chart": json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)})

    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale="RdBu", zmid=0,
            text=corr.values.round(2), texttemplate="%{text}",
            hovertemplate="%{x} vs %{y}: %{z:.3f}<extra></extra>"))
        fig = style_fig(fig, "Korrelyatsiya matrisasi", 350)
        charts.append({"title": "Korrelyatsiya matrisasi", "type": "heatmap",
                       "chart": json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)})

    if len(numeric_cols) >= 2:
        fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1],
                         color_discrete_sequence=["#f59e0b"], opacity=0.6)
        fig = style_fig(fig, f"Scatter: {numeric_cols[1]} vs {numeric_cols[0]}")
        charts.append({"title": f"Scatter: {numeric_cols[1]} vs {numeric_cols[0]}",
                       "type": "scatter",
                       "chart": json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)})

    if datetime_cols and numeric_cols:
        dt_col, num_col = datetime_cols[0], numeric_cols[0]
        try:
            tmp = df.copy()
            tmp[dt_col] = pd.to_datetime(tmp[dt_col], errors="coerce")
            tmp = tmp.dropna(subset=[dt_col, num_col]).sort_values(dt_col)
            if len(tmp) > 0:
                fig = px.line(tmp, x=dt_col, y=num_col, color_discrete_sequence=["#ec4899"])
                fig = style_fig(fig, f"Vaqt qatori: {num_col} / {dt_col}")
                charts.append({"title": f"Vaqt qatori: {num_col} / {dt_col}", "type": "line",
                               "chart": json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)})
        except:
            pass

    if numeric_cols:
        cols_box = numeric_cols[:6]
        fig = go.Figure()
        for c in cols_box:
            fig.add_trace(go.Box(y=df[c].dropna(), name=c))
        fig = style_fig(fig, "Box: sonli ustunlar", 320)
        charts.append({"title": "Box: sonli ustunlar", "type": "box",
                       "chart": json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)})

    return charts


# ========================= ROUTES =========================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        if "file" not in request.files:
            return jsonify({"error": "Fayl topilmadi"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Fayl tanlanmagan"}), 400
        filename = file.filename
        ext = os.path.splitext(filename)[1].lower()
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], f"{uuid.uuid4().hex}_{filename}")
        file.save(filepath)

        if ext == ".csv":
            for enc in ["utf-8", "latin-1", "cp1252", "utf-16"]:
                try:
                    df = pd.read_csv(filepath, encoding=enc)
                    break
                except:
                    continue
            else:
                df = pd.read_csv(filepath)
        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(filepath)
        elif ext == ".json":
            df = pd.read_json(filepath)
        elif ext == ".tsv":
            df = pd.read_csv(filepath, sep="\t")
        else:
            os.remove(filepath)
            return jsonify({"error": f"Format qollab-quvvatlanmaydi: {ext}. CSV, Excel, JSON qiling"}), 400

        if df.empty:
            os.remove(filepath)
            return jsonify({"error": "Fayl bosh yoki malumot yoq"}), 400

        session_id = uuid.uuid4().hex
        SESSIONS[session_id] = {
            "df": df,
            "original_df": df.copy(),
            "filename": filename,
            "filepath": filepath,
            "created_at": datetime.now().isoformat(),
            "history": []
        }
        return jsonify({
            "session_id": session_id,
            "filename": filename,
            "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
            "columns": list(df.columns),
            "profiling": generate_data_profiling(df),
            "preview": generate_preview(df),
            "auto_charts": generate_auto_charts(df)
        })
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/profile", methods=["POST"])
def get_profiling():
    try:
        sid = request.json.get("session_id")
        if sid not in SESSIONS:
            return jsonify({"error": "Sessiya topilmadi"}), 400
        return jsonify({"profiling": generate_data_profiling(SESSIONS[sid]["df"])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/preview", methods=["POST"])
def get_preview():
    try:
        d = request.json
        sid = d.get("session_id")
        if sid not in SESSIONS:
            return jsonify({"error": "Sessiya topilmadi"}), 400
        return jsonify({"preview": generate_preview(SESSIONS[sid]["df"], d.get("page", 1), d.get("per_page", 50))})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/clean", methods=["POST"])
def clean_data():
    try:
        d = request.json
        sid = d.get("session_id")
        if sid not in SESSIONS:
            return jsonify({"error": "Sessiya topilmadi"}), 400
        df = SESSIONS[sid]["df"].copy()
        operations = d.get("operations", [])
        results = []
        for op in operations:
            t = op.get("type")
            if t == "drop_duplicates":
                before = len(df); df = df.drop_duplicates()
                results.append(f"Dublikatlar ochirildi: {before - len(df)} ta qator")
            elif t == "drop_missing":
                thr = op.get("threshold", 0.5)
                bc = df.shape[1]; df = df.loc[:, df.isnull().mean() <= thr]
                dc = bc - df.shape[1]; br = len(df); df = df.dropna()
                results.append(f"Yetishmayotgan tozalandi: {dc} ustun, {br - len(df)} qator")
            elif t == "fill_missing":
                strat = op.get("strategy", "mean"); filled = 0
                for col in df.columns:
                    if df[col].isnull().any():
                        if pd.api.types.is_numeric_dtype(df[col]):
                            v = df[col].mean() if strat == "mean" else df[col].median() if strat == "median" else 0
                        else:
                            v = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                        filled += df[col].isnull().sum(); df[col] = df[col].fillna(v)
                results.append(f"Yetishmayotgan toldirildi: {filled} ta ({strat})")
            elif t == "remove_outliers":
                m = op.get("method", "iqr"); removed = 0
                for col in df.select_dtypes(include=[np.number]).columns:
                    if m == "iqr":
                        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                        iqr = q3 - q1; lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
                        mask = (df[col] >= lo) & (df[col] <= hi)
                        removed += (~mask).sum(); df = df[mask]
                    elif m == "zscore":
                        z = np.abs(stats.zscore(df[col].dropna()))
                        mask = z < op.get("threshold", 3)
                        vi = df[col].dropna().index[mask]
                        df = df.loc[df.index.isin(vi) | df[col].isnull()]
                results.append(f"Anomal qiymatlar: {removed} ta ({m})")
            elif t == "normalize":
                m = op.get("method", "minmax")
                for col in df.select_dtypes(include=[np.number]).columns:
                    if m == "minmax":
                        lo, hi = df[col].min(), df[col].max()
                        if hi - lo > 0: df[col] = (df[col] - lo) / (hi - lo)
                    elif m == "standard":
                        mu, s = df[col].mean(), df[col].std()
                        if s > 0: df[col] = (df[col] - mu) / s
                results.append(f"Normalizatsiya: {m}")
            elif t == "encode_categorical":
                m = op.get("method", "label")
                for col in df.select_dtypes(include=["object", "category"]).columns:
                    if m == "label":
                        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
                    elif m == "onehot":
                        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                        df = pd.concat([df.drop(col, axis=1), dummies], axis=1)
                results.append(f"Kategoriyali kodlandi: {m}")
            elif t == "drop_column":
                c = op.get("column")
                if c in df.columns: df = df.drop(c, axis=1); results.append(f"Ustun ochirildi: {c}")
            elif t == "rename_column":
                o, n = op.get("old_name"), op.get("new_name")
                if o in df.columns: df = df.rename(columns={o: n}); results.append(f"Renamed: {o} -> {n}")
            elif t == "change_type":
                c, nt = op.get("column"), op.get("new_type")
                if c in df.columns:
                    try:
                        if nt == "int": df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
                        elif nt == "float": df[c] = pd.to_numeric(df[c], errors="coerce")
                        elif nt == "str": df[c] = df[c].astype(str)
                        elif nt == "datetime": df[c] = pd.to_datetime(df[c], errors="coerce")
                        results.append(f"Tur ozgartirildi: {c} -> {nt}")
                    except Exception as ex:
                        results.append(f"Xato ({c}): {ex}")
        SESSIONS[sid]["df"] = df
        SESSIONS[sid]["history"].append({"action": "clean", "operations": operations, "results": results,
                                         "timestamp": datetime.now().isoformat()})
        return jsonify({
            "success": True, "results": results,
            "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
            "profiling": generate_data_profiling(df),
            "preview": generate_preview(df),
            "auto_charts": generate_auto_charts(df)
        })
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/chart", methods=["POST"])
def create_chart():
    try:
        d = request.json
        sid = d.get("session_id")
        if sid not in SESSIONS:
            return jsonify({"error": "Sessiya topilmadi"}), 400
        df = SESSIONS[sid]["df"]
        ct = d.get("chart_type")
        x, y, z = d.get("x_col"), d.get("y_col"), d.get("z_col")
        cc, sc = d.get("color_col"), d.get("size_col")
        af = d.get("agg_func", "sum")
        title = d.get("title", "")
        fig = None
        if ct == "scatter":
            fig = px.scatter(df, x=x, y=y, color=cc, size=sc, title=title or f"{y} vs {x}", template="plotly_dark")
        elif ct == "line":
            fig = px.line(df, x=x, y=y, color=cc, title=title or f"{y} over {x}", template="plotly_dark")
        elif ct == "bar":
            if af != "none" and y:
                g = df.groupby(x)[y].agg(af).reset_index()
                fig = px.bar(g, x=x, y=y, color=cc, title=title or f"{y} by {x} ({af})", template="plotly_dark")
            else:
                fig = px.bar(df, x=x, y=y, color=cc, title=title or f"Bar: {x}", template="plotly_dark")
        elif ct == "histogram":
            fig = px.histogram(df, x=x, color=cc, title=title or f"Distribution: {x}", template="plotly_dark")
        elif ct == "box":
            fig = px.box(df, x=x, y=y, color=cc, title=title or f"Box: {y} by {x}", template="plotly_dark")
        elif ct == "violin":
            fig = px.violin(df, x=x, y=y, color=cc, box=True, title=title or f"Violin: {y} by {x}", template="plotly_dark")
        elif ct == "pie":
            if y and af != "none":
                g = df.groupby(x)[y].agg(af).reset_index()
            else:
                g = df[x].value_counts().reset_index()
                g.columns = [x, "count"]
            fig = px.pie(g, names=x, values=g.columns[-1], title=title or f"Pie: {x}", template="plotly_dark")
        elif ct == "heatmap":
            ndf = df.select_dtypes(include=[np.number])
            if len(ndf.columns) > 0:
                corr = ndf.corr()
                fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns,
                    colorscale="RdBu", zmid=0, text=corr.values.round(3), texttemplate="%{text}",
                    hovertemplate="%{x} vs %{y}: %{z:.3f}<extra></extra>"))
                fig.update_layout(title=title or "Korrelyatsiya", template="plotly_dark")
        elif ct == "scatter3d":
            fig = px.scatter_3d(df, x=x, y=y, z=z, color=cc, title=title or f"3D", template="plotly_dark")
        elif ct == "area":
            fig = px.area(df, x=x, y=y, color=cc, title=title or f"Area", template="plotly_dark")
        elif ct == "sunburst":
            path = [x, cc] if cc else [x]
            fig = px.sunburst(df, path=path, values=y if y else None, title=title or f"Sunburst", template="plotly_dark")
        elif ct == "treemap":
            path = [x, cc] if cc else [x]
            fig = px.treemap(df, path=path, values=y if y else None, title=title or f"Treemap", template="plotly_dark")
        else:
            return jsonify({"error": f"Noma'lum grafik: {ct}"}), 400
        if fig:
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="#e0e0e0", size=12), margin=dict(l=40, r=20, t=50, b=40))
            return jsonify({"chart": json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)})
        return jsonify({"error": "Grafik yaratilmadi"}), 400
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/ml/predict", methods=["POST"])
def ml_predict():
    try:
        d = request.json
        sid = d.get("session_id")
        if sid not in SESSIONS:
            return jsonify({"error": "Sessiya topilmadi"}), 400
        df = SESSIONS[sid]["df"].copy()
        target = d.get("target")
        model_type = d.get("model_type", "auto")
        features = d.get("features", [])
        test_size = d.get("test_size", 0.2)
        ptype = d.get("problem_type", "auto")

        if target not in df.columns:
            return jsonify({"error": f"Maqsad ustun topilmadi: {target}"}), 400

        if ptype == "auto":
            ptype = "classification" if (df[target].dtype == "object" or df[target].nunique() <= 20) else "regression"

        X = df[features].copy() if features else df.drop(columns=[target]).copy()
        y = df[target].copy()

        for col in X.columns:
            if X[col].dtype == "object" or str(X[col].dtype) == "category":
                X[col] = X[col].astype(str)
                if X[col].nunique() > 50:
                    freq = X[col].value_counts().to_dict()
                    X[col] = X[col].map(freq)
                else:
                    X[col] = LabelEncoder().fit_transform(X[col])

        if ptype == "classification":
            if y.dtype == "object" or str(y.dtype) == "category":
                le_y = LabelEncoder()
                y = le_y.fit_transform(y.astype(str))
                class_names = list(le_y.classes_)
            else:
                class_names = sorted(y.unique())

        X = X.fillna(X.select_dtypes(include=[np.number]).median())
        y = y.fillna(y.median() if y.dtype in [np.float64, np.int64] else y.mode()[0])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        if ptype == "regression":
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
            }
            if model_type != "auto":
                m_map = {"linear": "Linear Regression", "random_forest": "Random Forest",
                         "gradient_boosting": "Gradient Boosting", "decision_tree": "Decision Tree"}
                model_name = m_map.get(model_type, "Random Forest")
                models = {model_name: models[model_name]}

            best_model, best_score, best_name, all_res = None, -np.inf, "", {}
            for name, mdl in models.items():
                try:
                    mdl.fit(X_train, y_train)
                    sc = mdl.score(X_test, y_test)
                    all_res[name] = {"r2": float(sc)}
                    if sc > best_score:
                        best_score, best_model, best_name = sc, mdl, name
                except:
                    continue
            model, model_name = best_model, best_name
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

            fi = {}
            if hasattr(model, "feature_importances_"):
                imps = model.feature_importances_
            elif hasattr(model, "coef_"):
                imps = np.abs(model.coef_)
            else:
                imps = np.zeros(len(X.columns))
            for i, c in enumerate(X.columns):
                fi[c] = float(imps[i]) if i < len(imps) else 0
            fi = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))

            cv = cross_val_score(model, X, y, cv=5, scoring="r2")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers", name="Bashorat vs Haqiqiy",
                                     marker=dict(color="#6366f1", size=6, opacity=0.6)))
            lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
            fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Ideal",
                                     line=dict(color="#ef4444", dash="dash")))
            fig.update_layout(title="Bashorat vs Haqiqiy", xaxis_title="Haqiqiy", yaxis_title="Bashorat",
                              template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
                              paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e0e0e0"))
            pred_chart = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

            cols = list(fi.keys())[:15]
            vals = [fi[c] for c in cols]
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=vals, y=cols, orientation="h", marker=dict(color="#6366f1")))
            fig2.update_layout(title="Feature Importance", xaxis_title="Muhimlik", yaxis_title="Ustun",
                               template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
                               paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e0e0e0"), margin=dict(l=120))
            fig2.update_yaxes(autorange="reversed")
            imp_chart = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

            return jsonify({
                "success": True, "problem_type": ptype, "model_name": model_name,
                "metrics": {"r2": float(r2), "rmse": rmse, "cv_mean": float(cv.mean()), "cv_std": float(cv.std())},
                "all_models": all_res, "feature_importance": fi,
                "prediction_chart": pred_chart, "importance_chart": imp_chart,
                "n_features": len(X.columns), "n_train": len(X_train), "n_test": len(X_test)
            })
        else:
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
            }
            if model_type != "auto":
                m_map = {"logistic": "Logistic Regression", "random_forest": "Random Forest",
                         "gradient_boosting": "Gradient Boosting", "decision_tree": "Decision Tree"}
                model_name = m_map.get(model_type, "Random Forest")
                models = {model_name: models[model_name]}

            best_model, best_acc, best_name, all_res = None, -1, "", {}
            for name, mdl in models.items():
                try:
                    mdl.fit(X_train, y_train)
                    acc = accuracy_score(y_test, mdl.predict(X_test))
                    all_res[name] = {"accuracy": float(acc)}
                    if acc > best_acc:
                        best_acc, best_model, best_name = acc, mdl, name
                except:
                    continue
            model, model_name = best_model, best_name
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

            fi = {}
            if hasattr(model, "feature_importances_"):
                imps = model.feature_importances_
            elif hasattr(model, "coef_"):
                imps = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
            else:
                imps = np.zeros(len(X.columns))
            for i, c in enumerate(X.columns):
                fi[c] = float(imps[i]) if i < len(imps) else 0
            fi = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))
            cv = cross_val_score(model, X, y, cv=5, scoring="accuracy")

            cols = list(fi.keys())[:15]
            vals = [fi[c] for c in cols]
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=vals, y=cols, orientation="h", marker=dict(color="#10b981")))
            fig2.update_layout(title="Feature Importance", template="plotly_dark",
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                               font=dict(color="#e0e0e0"), margin=dict(l=120))
            fig2.update_yaxes(autorange="reversed")
            imp_chart = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

            classes = [k for k in report.keys() if k not in ["accuracy", "macro avg", "weighted avg"]]
            ml = ["precision", "recall", "f1-score"]
            fig3 = go.Figure()
            for metric in ml:
                fig3.add_trace(go.Bar(name=metric, x=classes, y=[report[c].get(metric, 0) for c in classes]))
            fig3.update_layout(barmode="group", title="Klassifikatsiya hisoboti", template="plotly_dark",
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e0e0e0"))
            report_chart = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)

            return jsonify({
                "success": True, "problem_type": ptype, "model_name": model_name,
                "class_names": [str(c) for c in class_names],
                "metrics": {"accuracy": float(accuracy), "cv_mean": float(cv.mean()), "cv_std": float(cv.std())},
                "all_models": all_res, "classification_report": report,
                "feature_importance": fi, "importance_chart": imp_chart,
                "report_chart": report_chart, "n_features": len(X.columns),
                "n_train": len(X_train), "n_test": len(X_test)
            })
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/ml/cluster", methods=["POST"])
def ml_cluster():
    try:
        d = request.json
        sid = d.get("session_id")
        if sid not in SESSIONS:
            return jsonify({"error": "Sessiya topilmadi"}), 400
        df = SESSIONS[sid]["df"].copy()
        k = d.get("n_clusters", 3)
        features = d.get("features", [])
        X = df[features].copy() if features else df.select_dtypes(include=[np.number]).copy()
        for col in X.columns:
            if X[col].dtype == "object":
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        X = X.fillna(X.median())
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_s)
        df["cluster"] = clusters
        sil = silhouette_score(X_s, clusters) if k > 1 else 0
        pca = PCA(n_components=min(3, X.shape[1]))
        X_pca = pca.fit_transform(X_s)
        if X_pca.shape[1] >= 3:
            fig = px.scatter_3d(x=X_pca[:, 0], y=X_pca[:, 1], z=X_pca[:, 2],
                                color=clusters.astype(str), title=f"Klasterlash (K={k})",
                                template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Set2)
        else:
            fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1] if X_pca.shape[1] > 1 else [0]*len(X_pca),
                             color=clusters.astype(str), title=f"Klasterlash (K={k})",
                             template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e0e0e0"))
        chart = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        distortions, K_range = [], range(1, min(11, len(df)))
        for kk in K_range:
            kmeans_k = KMeans(n_clusters=kk, random_state=42, n_init=10)
            kmeans_k.fit(X_s)
            distortions.append(kmeans_k.inertia_)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=list(K_range), y=distortions, mode="lines+markers",
                                  marker=dict(color="#6366f1", size=8)))
        fig2.update_layout(title="Elbow Method", xaxis_title="K", yaxis_title="Inertia",
                           template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
                           paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e0e0e0"))
        elbow_chart = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

        summary = {}
        for c in range(k):
            cd = df[df["cluster"] == c]
            entry = {"size": int(len(cd)), "percentage": float(len(cd) / max(len(df), 1) * 100)}
            for col in X.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    entry[f"{col}_mean"] = float(cd[col].mean())
            summary[f"Klaster {c}"] = entry
        SESSIONS[sid]["df"] = df
        return jsonify({
            "success": True, "n_clusters": k, "silhouette_score": float(sil),
            "chart": chart, "elbow_chart": elbow_chart, "cluster_summary": summary,
            "explained_variance": [float(v) for v in pca.explained_variance_ratio_]
        })
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/stats", methods=["POST"])
def statistics():
    try:
        d = request.json
        sid = d.get("session_id")
        if sid not in SESSIONS:
            return jsonify({"error": "Sessiya topilmadi"}), 400
        df = SESSIONS[sid]["df"]
        at = d.get("analysis_type")
        c1, c2 = d.get("col1"), d.get("col2")
        results = {}
        if at == "correlation":
            ndf = df.select_dtypes(include=[np.number])
            if len(ndf.columns) < 2:
                return jsonify({"error": "Kamida 2 sonli ustun kerak"}), 400
            corr = ndf.corr()
            fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns,
                colorscale="RdBu", zmid=0, text=corr.values.round(3), texttemplate="%{text}",
                hovertemplate="%{x} vs %{y}: %{z:.3f}<extra></extra>"))
            fig.update_layout(title="Korrelyatsiya", template="plotly_dark",
                             plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e0e0e0"))
            results["chart"] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            results["correlation_matrix"] = {str(k): {str(k2): float(v2) for k2, v2 in v.items()} for k, v in corr.to_dict().items()}
        elif at == "ttest":
            if c1 and c2 and c1 in df.columns and c2 in df.columns:
                st, pv = stats.ttest_ind(df[c1].dropna(), df[c2].dropna())
                results["t_statistic"] = float(st); results["p_value"] = float(pv)
                results["significant"] = bool(pv < 0.05)
        elif at == "distribution":
            if c1 and c1 in df.columns:
                cd = df[c1].dropna()
                if pd.api.types.is_numeric_dtype(cd):
                    s = cd[:5000] if len(cd) > 5000 else cd
                    if len(s) >= 3:
                        st, pv = stats.shapiro(s)
                        results["shapiro_stat"] = float(st); results["shapiro_p"] = float(pv)
                        results["normal"] = bool(pv > 0.05)
                    results["skewness"] = float(stats.skew(cd))
                    results["kurtosis"] = float(stats.kurtosis(cd))
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=cd, name="Haqiqiy", opacity=0.7, marker_color="#6366f1"))
                    mu, s2 = cd.mean(), cd.std()
                    xr = np.linspace(cd.min(), cd.max(), 100)
                    if s2 > 0:
                        pdf = stats.norm.pdf(xr, mu, s2) * len(cd) * (cd.max() - cd.min()) / 30
                        fig.add_trace(go.Scatter(x=xr, y=pdf, name="Normal", line=dict(color="#ef4444", width=2)))
                    fig.update_layout(title=f"Taqsimot: {c1}", template="plotly_dark",
                                     plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e0e0e0"))
                    results["chart"] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return jsonify({"success": True, "results": results})
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/ai_insights", methods=["POST"])
def ai_insights():
    try:
        d = request.json
        sid = d.get("session_id")
        if sid not in SESSIONS:
            return jsonify({"error": "Sessiya topilmadi"}), 400
        df = SESSIONS[sid]["df"]
        insights = []
        nc = df.select_dtypes(include=[np.number]).columns.tolist()
        cc = df.select_dtypes(include=["object", "category"]).columns.tolist()
        insights.append({"type": "info", "icon": "database", "title": "Malumot hajmi",
            "text": f"Dataset {df.shape[0]:,} qator va {df.shape[1]} ustun. {len(nc)} sonli, {len(cc)} kategoriyali."})
        missing = df.isnull().sum()
        if missing.sum() > 0:
            wc = missing.idxmax(); wp = missing.max() / max(len(df), 1) * 100
            insights.append({"type": "warning", "icon": "alert-triangle", "title": "Yetishmayotgan malumot",
                "text": f"Jami {missing.sum():,} yetishmayotgan. Eng yomon: '{wc}' ({wp:.1f}%)."})
        else:
            insights.append({"type": "success", "icon": "check-circle", "title": "Malumot to'liqligi",
                "text": "Yetishmayotgan qiymatlar yoq."})
        dup = df.duplicated().sum()
        if dup > 0:
            insights.append({"type": "warning", "icon": "copy", "title": "Dublikatlar",
                "text": f"{dup} dublikat qator ({dup/max(len(df),1)*100:.1f}%)."})
        if len(nc) >= 2:
            corr = df[nc].corr()
            pairs = []
            for i in range(len(nc)):
                for j in range(i+1, len(nc)):
                    pairs.append((nc[i], nc[j], abs(corr.iloc[i, j]), corr.iloc[i, j]))
            pairs.sort(key=lambda x: x[2], reverse=True)
            if pairs and pairs[0][2] > 0.5:
                p = pairs[0]
                d2 = "musbat" if p[3] > 0 else "manfiy"
                s2 = "kuchli" if p[2] > 0.7 else "o'rtacha"
                insights.append({"type": "info", "icon": "trending-up", "title": "Korrelyatsiya",
                    "text": f"{s2} {d2}: '{p[0]}' va '{p[1]}' (r={p[3]:.3f})."})
        for col in cc:
            if df[col].nunique() > 50:
                insights.append({"type": "warning", "icon": "layers", "title": f"Yuqori kardinallik: {col}",
                    "text": f"'{col}' ustunida {df[col].nunique()} noyob qiymat."})
                break
        oc = 0
        for col in nc:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            oc += ((df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)).sum()
        if oc > 0:
            insights.append({"type": "warning", "icon": "zap", "title": "Anomal qiymatlar",
                "text": f"IQR bo'yicha ~{oc} anomal qiymat."})
        for col in nc:
            sk = df[col].skew()
            if abs(sk) > 2:
                d2 = "o'ng" if sk > 0 else "chap"
                insights.append({"type": "info", "icon": "bar-chart", "title": f"Noto'g'ri taqsimot: {col}",
                    "text": f"'{col}' {d2} tomonga og'gan (skew={sk:.2f})."})
                break
        if len(nc) >= 2:
            best_t, best_r = None, -np.inf
            for tgt in nc:
                feats = [c for c in nc if c != tgt]
                if not feats:
                    continue
                try:
                    td = df[feats + [tgt]].dropna()
                    if len(td) < 10:
                        continue
                    m = LinearRegression()
                    cv = cross_val_score(m, td[feats], td[tgt], cv=min(5, len(td)//2), scoring="r2")
                    if cv.mean() > best_r:
                        best_r, best_t = cv.mean(), tgt
                except:
                    continue
            if best_t and best_r > 0.3:
                insights.append({"type": "success", "icon": "target", "title": "ML tavsiyasi",
                    "text": f"'{best_t}' bashorat qilish uchun yaxshi (R2~{best_r:.2f})."})
        return jsonify({"insights": insights})
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/download", methods=["POST"])
def download_data():
    try:
        d = request.json
        sid = d.get("session_id")
        fmt = d.get("format", "csv")
        if sid not in SESSIONS:
            return jsonify({"error": "Sessiya topilmadi"}), 400
        df = SESSIONS[sid]["df"]
        fn = SESSIONS[sid]["filename"].rsplit(".", 1)[0]
        fp = os.path.join(app.config["UPLOAD_FOLDER"], f"{fn}_cleaned.{fmt}")
        if fmt == "csv": df.to_csv(fp, index=False)
        elif fmt == "xlsx": df.to_excel(fp, index=False)
        elif fmt == "json": df.to_json(fp, orient="records", indent=2)
        return send_file(fp, as_attachment=True, download_name=f"{fn}_cleaned.{fmt}")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/columns", methods=["POST"])
def get_columns():
    try:
        d = request.json
        sid = d.get("session_id")
        if sid not in SESSIONS:
            return jsonify({"error": "Sessiya topilmadi"}), 400
        df = SESSIONS[sid]["df"]
        info = []
        for col in df.columns:
            info.append({
                "name": col, "dtype": str(df[col].dtype),
                "n_unique": int(df[col].nunique()), "n_missing": int(df[col].isnull().sum()),
                "is_numeric": pd.api.types.is_numeric_dtype(df[col]),
                "is_categorical": df[col].dtype == "object" or str(df[col].dtype) == "category",
                "sample_values": df[col].dropna().head(3).tolist()
            })
        return jsonify({"columns": info})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
