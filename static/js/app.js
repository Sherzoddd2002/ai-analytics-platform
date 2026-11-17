let sessionId = null;
let currentColumns = [];
let currentPage = 1;
let totalPages = 1;

// Escape untrusted values coming from uploaded files before inserting into HTML.
function escapeHtml(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function renderStatsCards(p) {
    const cards = [
        { label: 'Qatorlar', value: Number(p.shape.rows).toLocaleString() },
        { label: 'Ustunlar', value: p.shape.cols },
        { label: 'Yetishmayotgan', value: Number(p.quality.total_missing).toLocaleString() },
        { label: 'Dublikatlar', value: p.quality.duplicates },
        { label: 'To\'liq qatorlar', value: Number(p.quality.complete_rows).toLocaleString() },
        { label: 'Xotira', value: p.memory_usage },
    ];
    document.getElementById('stats-cards').innerHTML = cards.map(c =>
        `<div class="stat-card"><div class="label">${escapeHtml(c.label)}</div><div class="value">${escapeHtml(c.value)}</div></div>`
    ).join('');
}

// A server restart drops in-memory sessions. Reset the UI instead of leaving
// the user stuck with dead buttons and "Sessiya topilmadi" toasts.
function resetSession(message) {
    sessionId = null;
    currentColumns = [];
    document.querySelectorAll('.nav-btn').forEach(b => {
        if (b.dataset.tab !== 'upload') b.disabled = true;
    });
    document.getElementById('file-info').classList.add('hidden');
    document.getElementById('stats-cards').innerHTML = '';
    document.getElementById('auto-charts').innerHTML = '';
    ['clean-results', 'ml-results', 'cluster-results', 'stats-results', 'insights-container'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.innerHTML = '';
    });
    switchTab('upload', true);
    showToast(message || 'Sessiya tugadi. Faylni qaytadan yuklang.', 'error');
}

function handleDataError(data) {
    if (!data || !data.error) return false;
    if (data.session_expired || String(data.error).includes('Sessiya topilmadi')) {
        resetSession('Sessiya tugadi (server qayta ishga tushgan). Faylni qaytadan yuklang.');
    } else {
        showToast(data.error, 'error');
    }
    return true;
}

// ===== Upload =====
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', e => { if (e.target.files.length) handleFile(e.target.files[0]); });

function handleFile(file) {
    document.getElementById('upload-error').classList.add('hidden');
    document.getElementById('upload-progress').classList.remove('hidden');
    const formData = new FormData();
    formData.append('file', file);
    fetch('/upload', { method: 'POST', body: formData })
        .then(r => r.json())
        .then(data => {
            document.getElementById('upload-progress').classList.add('hidden');
            if (data.error) {
                document.getElementById('upload-error').textContent = data.error;
                document.getElementById('upload-error').classList.remove('hidden');
                return;
            }
            sessionId = data.session_id;
            currentColumns = data.columns;
            document.getElementById('file-info').innerHTML = `<b>${escapeHtml(data.filename)}</b><br>${data.shape.rows} qator × ${data.shape.cols} ustun`;
            document.getElementById('file-info').classList.remove('hidden');
            document.querySelectorAll('.nav-btn').forEach(b => b.disabled = false);
            showToast(data.note || 'Fayl muvaffaqiyatli yuklandi!', data.note ? 'info' : 'success');
            renderDashboard(data);
            populateSelectors(data.columns);
            // Data is already in the upload response: don't re-fetch it.
            switchTab('dashboard', true);
        })
        .catch(err => {
            document.getElementById('upload-progress').classList.add('hidden');
            document.getElementById('upload-error').textContent = 'Xato: ' + err.message;
            document.getElementById('upload-error').classList.remove('hidden');
        });
}

// ===== Navigation =====
document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        if (btn.disabled) return;
        switchTab(btn.dataset.tab);
    });
});

function switchTab(tab, skipRefresh = false) {
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    document.querySelector(`.nav-btn[data-tab="${tab}"]`).classList.add('active');
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    document.getElementById(`tab-${tab}`).classList.add('active');
    if (skipRefresh || !sessionId) return;
    if (tab === 'dashboard') refreshDashboard();
    if (tab === 'data') loadPreview(currentPage);
    if (tab === 'clean') refreshClean();
    if (tab === 'insights') loadInsights();
}

// ===== Dashboard =====
function renderDashboard(data) {
    renderStatsCards(data.profiling);
    renderCharts(data.auto_charts);
}

function refreshDashboard() {
    // Server-side cache makes this a few KB instead of several MB.
    fetch('/profile', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ session_id: sessionId })
    }).then(r => r.json()).then(data => {
        if (handleDataError(data)) return;
        renderStatsCards(data.profiling);
    });
}

function refreshCharts() {
    return fetch('/charts/auto', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ session_id: sessionId })
    }).then(r => r.json()).then(data => {
        if (handleDataError(data)) return;
        renderCharts(data.auto_charts);
    });
}

function renderCharts(charts) {
    if (charts) {
        const container = document.getElementById('auto-charts');
        container.innerHTML = charts.map(c =>
            `<div class="chart-card"><div class="chart-title">${escapeHtml(c.title)}</div><div class="chart-plot"></div></div>`
        ).join('');
        // Parse once per chart and use static mode: much cheaper for Plotly.
        container.querySelectorAll('.chart-card').forEach((card, i) => {
            const parsed = JSON.parse(charts[i].chart);
            // The card already shows the title; don't draw it twice inside the plot.
            if (parsed.layout && parsed.layout.title) parsed.layout.title = '';
            Plotly.newPlot(card.querySelector('.chart-plot'), parsed.data, parsed.layout,
                           {responsive: true, displaylogo: false,
                            modeBarButtonsToRemove: ['lasso2d', 'select2d']});
        });
    }
}

// ===== Data Table =====
function refreshData() {
    loadPreview(1);
}

function loadPreview(page) {
    currentPage = page;
    fetch('/preview', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ session_id: sessionId, page, per_page: 50 })
    }).then(r => r.json()).then(data => {
        if (handleDataError(data)) return;
        const p = data.preview;
        totalPages = p.total_pages;
        let html = '<thead><tr>';
        p.columns.forEach(c => html += `<th>${escapeHtml(c)}</th>`);
        html += '</tr></thead><tbody>';
        p.data.forEach(row => {
            html += '<tr>';
            p.columns.forEach(c => {
                const val = row[c];
                html += `<td title="${escapeHtml(val)}">${val === null ? '<span style="color:#666">NaN</span>' : escapeHtml(val)}</td>`;
            });
            html += '</tr>';
        });
        html += '</tbody>';
        document.getElementById('data-table').innerHTML = html;
        document.getElementById('page-info').textContent = `${p.page} / ${p.total_pages} (${p.total} ta)`;
    });
}

document.getElementById('prev-page').addEventListener('click', () => { if (currentPage > 1) loadPreview(currentPage - 1); });
document.getElementById('next-page').addEventListener('click', () => { if (currentPage < totalPages) loadPreview(currentPage + 1); });

// ===== Clean =====
function refreshClean() {
    fetch('/columns', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ session_id: sessionId })
    }).then(r => r.json()).then(data => {
        if (handleDataError(data)) return;
        const dropSel = document.getElementById('drop-col');
        dropSel.innerHTML = data.columns.map(c =>
            `<option value="${escapeHtml(c.name)}">${escapeHtml(c.name)}</option>`).join('');
        renderProfiling(data.columns);
    });
}

function renderProfiling(columns) {
    const container = document.getElementById('profiling-display');
    container.innerHTML = columns.map(c => {
        let rows = '';
        rows += profRow('Tip', c.dtype);
        rows += profRow('Noyob', c.n_unique);
        rows += profRow('Yetishmayotgan', `${c.n_missing} (${(c.n_missing / (c.n_unique + c.n_missing || 1) * 100).toFixed(1)}%)`);
        rows += profRow('Sonli', c.is_numeric ? 'Ha' : 'Yoq');
        if (c.sample_values && c.sample_values.length)
            rows += profRow('Namuna', c.sample_values.map(v => String(v).substring(0, 30)).join(', '));
        return `<div class="prof-col"><h4>${escapeHtml(c.name)}</h4>${rows}</div>`;
    }).join('');
}

function profRow(label, value) {
    return `<div class="prof-row"><span>${escapeHtml(label)}</span><span>${escapeHtml(value ?? '-')}</span></div>`;
}

function runClean(operations) {
    fetch('/clean', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ session_id: sessionId, operations })
    }).then(r => r.json()).then(data => {
        if (handleDataError(data)) return;
        const container = document.getElementById('clean-results');
        container.innerHTML = data.results.map(r =>
            `<div class="clean-result-item">${escapeHtml(r)}</div>`).join('');
        showToast('Tozalash bajarildi!', 'success');
        refreshClean();
        renderStatsCards(data.profiling);
        refreshCharts(); // rebuild dashboard charts from the cleaned data
    });
}

function downloadData(format) {
    fetch('/download', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ session_id: sessionId, format })
    }).then(async r => {
        if (!r.ok) {
            const err = await r.json().catch(() => ({}));
            throw new Error(err.error || `HTTP ${r.status}`);
        }
        return r.blob();
    }).then(blob => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `cleaned_data.${format}`;
        a.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
        showToast('Yuklab olindi!', 'success');
    }).catch(err => showToast('Xato: ' + err.message, 'error'));
}

// ===== Charts =====
function populateSelectors(columns) {
    ['chart-x', 'chart-y', 'chart-z', 'chart-color', 'chart-size',
     'ml-target', 'stats-col1', 'stats-col2', 'drop-col', 'cluster-features'].forEach(id => {
        const el = document.getElementById(id);
        if (!el) return;
        const isMulti = el.multiple;
        const current = el.value;
        el.innerHTML = (isMulti ? '' : '<option value="">-- Tanlang --</option>') +
            columns.map(c => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`).join('');
        if (current) el.value = current;
    });
}

function createChart() {
    const payload = {
        session_id: sessionId,
        chart_type: document.getElementById('chart-type').value,
        x_col: document.getElementById('chart-x').value,
        y_col: document.getElementById('chart-y').value,
        z_col: document.getElementById('chart-z').value,
        color_col: document.getElementById('chart-color').value,
        size_col: document.getElementById('chart-size').value,
        agg_func: document.getElementById('chart-agg').value,
    };
    fetch('/chart', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
    }).then(r => r.json()).then(data => {
        if (handleDataError(data)) return;
        const chartData = JSON.parse(data.chart);
        Plotly.newPlot('custom-chart', chartData.data, chartData.layout, {responsive: true, displaylogo: false});
    });
}

// ===== ML =====
document.querySelectorAll('.ml-tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.ml-tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        document.querySelectorAll('.ml-section').forEach(s => s.classList.remove('active'));
        document.getElementById(`mltab-${btn.dataset.mltab}`).classList.add('active');
    });
});

function runMLPredict() {
    const payload = {
        session_id: sessionId,
        target: document.getElementById('ml-target').value,
        model_type: document.getElementById('ml-model').value,
        problem_type: document.getElementById('ml-problem').value,
        test_size: parseInt(document.getElementById('ml-test-size').value) / 100,
    };
    if (!payload.target) return showToast('Maqsad ustun tanlang!', 'error');
    document.getElementById('ml-results').innerHTML = '<div class="spinner"></div><p>Model o\'rgatilmoqda...</p>';
    fetch('/ml/predict', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
    }).then(r => r.json()).then(data => {
        if (handleDataError(data)) return;
        if (data.error) {
            document.getElementById('ml-results').innerHTML = `<div class="error-msg">${escapeHtml(data.error)}</div>`;
            return;
        }
        renderMLResults(data);
    });
}

function renderMLResults(data) {
    const m = data.metrics;
    let html = `<h3>${data.model_name}</h3>`;
    html += `<p style="color:var(--text-dim);margin:8px 0">${data.problem_type === 'classification' ? 'Klassifikatsiya' : 'Regressiya'} | ${data.n_train} train / ${data.n_test} test | ${data.n_features} xususiyat</p>`;
    if (data.problem_type === 'regression') {
        html += `<div class="ml-metric"><div class="label">R² Score</div><div class="value">${m.r2.toFixed(4)}</div></div>`;
        html += `<div class="ml-metric"><div class="label">RMSE</div><div class="value">${m.rmse.toFixed(4)}</div></div>`;
        html += `<div class="ml-metric"><div class="label">CV Mean</div><div class="value">${m.cv_mean.toFixed(4)}</div></div>`;
        html += `<div class="ml-metric"><div class="label">CV Std</div><div class="value">${m.cv_std.toFixed(4)}</div></div>`;
    } else {
        html += `<div class="ml-metric"><div class="label">Accuracy</div><div class="value">${(m.accuracy * 100).toFixed(2)}%</div></div>`;
        html += `<div class="ml-metric"><div class="label">CV Mean</div><div class="value">${(m.cv_mean * 100).toFixed(2)}%</div></div>`;
        html += `<div class="ml-metric"><div class="label">CV Std</div><div class="value">${(m.cv_std * 100).toFixed(2)}%</div></div>`;
    }
    // All models comparison
    if (data.all_models && Object.keys(data.all_models).length > 1) {
        html += '<h4 style="margin:16px 0 8px">Barcha Modellar:</h4>';
        for (const [name, res] of Object.entries(data.all_models)) {
            if (res.error) {
                html += `<div class="ml-metric"><div class="label">${escapeHtml(name)}</div><div class="value" style="font-size:14px">${escapeHtml(res.error)}</div></div>`;
                continue;
            }
            const v = res.r2 !== undefined ? `R²=${res.r2.toFixed(4)}` : `Acc=${(res.accuracy * 100).toFixed(2)}%`;
            html += `<div class="ml-metric"><div class="label">${escapeHtml(name)}</div><div class="value" style="font-size:16px">${v}</div></div>`;
        }
    }
    html += '<div id="ml-pred-chart" style="margin-top:16px"></div>';
    html += '<div id="ml-imp-chart" style="margin-top:16px"></div>';
    if (data.report_chart) html += '<div id="ml-report-chart" style="margin-top:16px"></div>';
    document.getElementById('ml-results').innerHTML = html;
    if (data.prediction_chart) {
        const cd = JSON.parse(data.prediction_chart);
        Plotly.newPlot('ml-pred-chart', cd.data, cd.layout, {responsive: true, displaylogo: false});
    }
    if (data.importance_chart) {
        const cd = JSON.parse(data.importance_chart);
        Plotly.newPlot('ml-imp-chart', cd.data, cd.layout, {responsive: true, displaylogo: false});
    }
    if (data.report_chart) {
        const cd = JSON.parse(data.report_chart);
        Plotly.newPlot('ml-report-chart', cd.data, cd.layout, {responsive: true, displaylogo: false});
    }
}

function runCluster() {
    const k = parseInt(document.getElementById('cluster-k').value);
    const feats = Array.from(document.getElementById('cluster-features').selectedOptions).map(o => o.value);
    document.getElementById('cluster-results').innerHTML = '<div class="spinner"></div><p>Klasterlash...</p>';
    fetch('/ml/cluster', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ session_id: sessionId, n_clusters: k, features: feats })
    }).then(r => r.json()).then(data => {
        if (handleDataError(data)) return;
        if (data.error) {
            document.getElementById('cluster-results').innerHTML = `<div class="error-msg">${escapeHtml(data.error)}</div>`;
            return;
        }
        let html = `<div class="ml-metric"><div class="label">Silhouette Score</div><div class="value">${data.silhouette_score.toFixed(4)}</div></div>`;
        html += '<div id="cluster-chart" style="margin-top:16px"></div>';
        html += '<div id="elbow-chart" style="margin-top:16px"></div>';
        html += '<h4 style="margin:16px 0 8px">Klaster Xulosasi:</h4>';
        for (const [name, info] of Object.entries(data.cluster_summary)) {
            html += `<div class="ml-metric"><div class="label">${name}</div><div class="value" style="font-size:16px">${info.size} (${info.percentage.toFixed(1)}%)</div></div>`;
        }
        document.getElementById('cluster-results').innerHTML = html;
        const cd = JSON.parse(data.chart);
        Plotly.newPlot('cluster-chart', cd.data, cd.layout, {responsive: true, displaylogo: false});
        const ed = JSON.parse(data.elbow_chart);
        Plotly.newPlot('elbow-chart', ed.data, ed.layout, {responsive: true, displaylogo: false});
    });
}

// ===== Stats =====
function runStats() {
    const payload = {
        session_id: sessionId,
        analysis_type: document.getElementById('stats-type').value,
        col1: document.getElementById('stats-col1').value,
        col2: document.getElementById('stats-col2').value,
    };
    fetch('/stats', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
    }).then(r => r.json()).then(data => {
        if (handleDataError(data)) return;
        const r = data.results;
        let html = '';
        if (r.chart) {
            html += '<div id="stats-chart" style="margin-bottom:16px"></div>';
        }
        if (r.t_statistic !== undefined) {
            html += `<div class="ml-metric"><div class="label">T-Statistic</div><div class="value">${r.t_statistic.toFixed(4)}</div></div>`;
            html += `<div class="ml-metric"><div class="label">P-Value</div><div class="value">${r.p_value.toFixed(4)}</div></div>`;
            html += `<div class="ml-metric"><div class="label">Ahamiyatli (p<0.05)</div><div class="value">${r.significant ? 'Ha' : 'Yoq'}</div></div>`;
        }
        if (r.skewness !== undefined) {
            html += `<div class="ml-metric"><div class="label">Skewness</div><div class="value">${r.skewness.toFixed(4)}</div></div>`;
            html += `<div class="ml-metric"><div class="label">Kurtosis</div><div class="value">${r.kurtosis.toFixed(4)}</div></div>`;
            if (r.shapiro_stat !== undefined) {
                html += `<div class="ml-metric"><div class="label">Shapiro Stat</div><div class="value">${r.shapiro_stat.toFixed(4)}</div></div>`;
                html += `<div class="ml-metric"><div class="label">Normal Taqsimot</div><div class="value">${r.normal ? 'Ha' : 'Yoq'}</div></div>`;
            }
        }
        document.getElementById('stats-results').innerHTML = html || '<p>Tahlil natijasi yoq</p>';
        if (r.chart) {
            const cd = JSON.parse(r.chart);
            Plotly.newPlot('stats-chart', cd.data, cd.layout, {responsive: true, displaylogo: false});
        }
    });
}

// ===== AI Insights =====
function loadInsights() {
    if (!sessionId) return;
    document.getElementById('insights-container').innerHTML = '<div class="spinner"></div>';
    fetch('/ai_insights', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ session_id: sessionId })
    }).then(r => r.json()).then(data => {
        if (handleDataError(data)) return;
        const icons = {
            'database': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>',
            'alert-triangle': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
            'check-circle': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
            'copy': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>',
            'trending-up': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>',
            'layers': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>',
            'zap': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>',
            'bar-chart': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>',
            'target': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
        };
        document.getElementById('insights-container').innerHTML = data.insights.map(i => `
            <div class="insight-card ${i.type}">
                <div class="insight-icon">${icons[i.icon] || icons['database']}</div>
                <div class="insight-content"><h4>${escapeHtml(i.title)}</h4><p>${escapeHtml(i.text)}</p></div>
            </div>
        `).join('');
    });
}

// ===== Toast =====
function showToast(msg, type) {
    const t = document.getElementById('toast');
    t.textContent = msg;
    t.className = 'toast ' + (type || '');
    setTimeout(() => t.classList.add('hidden'), 3000);
}
