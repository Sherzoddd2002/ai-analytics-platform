# AI Analytics Platform

A professional AI analytics platform in the style of Julius.ai. Upload a file and get automatic analysis, cleaning, charts, and ML predictions — all in one place. The interface and all user-facing messages are in English.

## Features

| Section | Function |
|---------|----------|
| File Upload | CSV, Excel (.xlsx, .xls), JSON, TSV |
| Dashboard | Automatic statistics and auto-generated charts |
| Data Cleaning | Duplicates, missing values, outliers (IQR/Z-score), normalization, categorical encoding |
| Charts | Scatter, Line, Bar, Histogram, Box, Violin, Pie, Area, Heatmap, 3D Scatter, Sunburst, Treemap (12 types) |
| Machine Learning | Regression (Linear, Random Forest, Gradient Boosting, Decision Tree), classification, automatic model selection, feature importance |
| Clustering | K-Means, Elbow method, 3D PCA visualization |
| Statistical Analysis | Correlation matrix, T-test, distribution analysis (Shapiro-Wilk, skewness, kurtosis) |
| AI Insights | Automatic data analysis and recommendations |
| Download | Export cleaned data as CSV/Excel/JSON |

## Requirements

- **Python 3.10+** (3.12 or 3.13 recommended)
- **pip** (included with Python)

## Installation (step by step)

### 1. Get the project

```bash
git clone https://github.com/Sherzoddd2002/ai-analytics-platform.git
cd ai-analytics-platform
```

Or download the ZIP file and extract it.

### 2. Create a virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
python app.py
```

The app uses **Waitress** as its WSGI server when it is installed; otherwise it falls back to the Flask development server. Waitress starts silently — no "Running on http://..." banner is printed.

### 5. Open it in your browser

```
http://localhost:5000
```

## How to use

1. **Upload a file** — drag and drop a CSV or Excel file
2. **Dashboard** — view automatic charts and statistics
3. **Clean** — remove duplicates, handle NaN values, outliers, and encoding
4. **Charts** — build your own chart (12 types)
5. **ML / Prediction** — choose a target column and run a prediction (auto-model)
6. **Clustering** — group rows with K-Means
7. **Statistics** — run correlation, T-test, and distribution analysis
8. **AI Insights** — get automatic recommendations
9. **Download** — export the cleaned data

## Tech stack

- **Backend**: Flask 3.1, Pandas 2.2, NumPy 2.1, SciPy 1.15, scikit-learn 1.6, statsmodels 0.14
- **Frontend**: HTML5, CSS3, vanilla JavaScript, Plotly.js (bundled locally at `static/js/plotly.min.js`, no CDN)
- **Visualization**: Plotly

## Project structure

```
ai-analytics-platform/
|-- app.py                 # Flask backend and all API endpoints
|-- requirements.txt       # Dependencies
|-- README.md              # Documentation
|-- .gitignore
|-- templates/
|   `-- index.html         # Main page
|-- static/
|   |-- css/
|   |   `-- style.css      # Styles
|   `-- js/
|       |-- app.js         # Frontend logic
|       `-- plotly.min.js  # Bundled Plotly.js (offline-safe)
`-- uploads/               # Uploaded files (created automatically)
```

## Troubleshooting

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

If port 5000 is already in use, set the `PORT` environment variable:

```bash
# Windows
set PORT=5055 && python app.py
# macOS / Linux
PORT=5055 python app.py
```

If you see an expired-session message after a server restart, simply upload your file again. The UI resets automatically instead of leaving dead buttons behind.

## Performance optimizations

- Profiling, charts, column information, and AI Insights are **computed once and cached per session**; switching tabs does not recompute them.
- Charts use a **2,000-row sample** (a 100,000-row scatter chart is now about 0.24 MB gzipped instead of 22 MB).
- ML: models are trained on a 10,000-row sample with 3-fold cross-validation on 3,000 rows; the best model is **not retrained a second time**.
- Clustering: 10,000-row sample, `MiniBatchKMeans` for large datasets, silhouette on 2,000 points, elbow method limited to 8 clusters.
- `/clean` now returns only results plus a small profile/preview; charts are fetched separately from `/charts/auto`.
- JSON responses are **gzipped**; `debug=False`; sessions are limited to 6 hours and 20 concurrent sessions.
- Plotly.js is **bundled locally** (`static/js/plotly.min.js`), so blocked CDNs no longer stall the page for 25 seconds.
- The first Plotly chart creation (~7 seconds) is **warmed up in the background at startup**; early uploads wait for that warmup, later uploads take about 2 seconds.
- `OMP_NUM_THREADS=1` prevents Windows `wmic` errors and MKL slowdowns.
- File names and user data are HTML-escaped, tracebacks are not sent to the browser, and limits of 250,000 rows / 200 columns are enforced.

Charts, AI Insights, and ML predictions are computed from samples, so they are approximate indicators. Data cleaning and export always run on the full dataset.

## License

MIT — free to use.
