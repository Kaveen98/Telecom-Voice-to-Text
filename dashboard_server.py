"""
dashboard_server.py
Optional Flask dashboard for the Telecom Voice-to-Text realtime pipeline.

Usage:
    python dashboard_server.py              # opens on http://127.0.0.1:5050
    python dashboard_server.py --port 8080  # custom port

Primary runtime is watcher.py. For production, run this dashboard only behind
proper access controls if remote access is required.

The dashboard auto-refreshes every 10 seconds and shows:
  - Calls transcribed today
  - Cost today (USD and LKR)
  - Tokens used (audio, text, output)
  - Silence stripped today (time saved + cost saved)
  - Language breakdown (Sinhala / English / Tamil)
  - Realtime metadata rows, with archived-mode rows shown separately
  - 14-day cost trend
  - Last 20 calls with full detail

Requires:
    python -m pip install -r requirements.txt
"""
from __future__ import annotations

import argparse
import csv
import functools
import io
import json
import os
import re
import secrets
import sys
from pathlib import Path

try:
    from flask import (
        Flask, Response, abort, jsonify, redirect, render_template_string,
        request, send_file, session, url_for,
    )
    from werkzeug.security import check_password_hash
except ImportError:
    print("ERROR: Flask is not installed.")
    print("Run:  python -m pip install -r requirements.txt")
    sys.exit(1)

from config import TRANSCRIPT_OUTPUT_DIR
from database import get_call_transcript, get_dashboard_data
from transcript_storage import resolve_transcript_path

app = Flask(__name__)

# ── Secret key (persisted so sessions survive restarts) ───────────────────────
_SECRET_FILE = Path(__file__).parent / ".dashboard_secret"

def _load_secret() -> str:
    env_key = os.getenv("DASHBOARD_SECRET_KEY", "").strip()
    if env_key:
        return env_key
    if _SECRET_FILE.exists():
        return _SECRET_FILE.read_text().strip()
    key = secrets.token_hex(32)
    _SECRET_FILE.write_text(key)
    return key

app.secret_key = _load_secret()

# ── User store (users.json) ───────────────────────────────────────────────────
_USERS_FILE = Path(__file__).parent / "users.json"

def _load_users() -> list[dict]:
    if not _USERS_FILE.exists():
        return []
    try:
        return json.loads(_USERS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

def _find_user(username: str) -> dict | None:
    for u in _load_users():
        if u.get("username", "").lower() == username.lower():
            return u
    return None

# ── Login required decorator ──────────────────────────────────────────────────
def login_required(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("user"):
            return redirect(url_for("login", next=request.path))
        return f(*args, **kwargs)
    return decorated

# ── Login page HTML ───────────────────────────────────────────────────────────
_LOGIN_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Login — SLT Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', Arial, sans-serif;
    background: #f0f4f8;
    display: flex; align-items: center; justify-content: center;
    min-height: 100vh;
  }
  .card {
    background: #fff;
    border-radius: 12px;
    box-shadow: 0 4px 24px rgba(0,0,0,.10);
    padding: 40px 36px;
    width: 100%; max-width: 380px;
  }
  .logo { text-align: center; margin-bottom: 28px; }
  .logo img {
    display: block; width: 84px; height: 84px; object-fit: contain;
    margin: 0 auto 14px; border-radius: 16px;
  }
  .logo h1 { font-size: 20px; font-weight: 700; color: #1a73e8; }
  .logo p  { font-size: 13px; color: #666; margin-top: 4px; }
  label { display: block; font-size: 13px; font-weight: 600;
          color: #444; margin-bottom: 5px; }
  input[type=text], input[type=password] {
    width: 100%; padding: 10px 12px; border: 1px solid #d0d5dd;
    border-radius: 7px; font-size: 14px; outline: none;
    transition: border-color .15s;
  }
  input:focus { border-color: #1a73e8; }
  .field { margin-bottom: 18px; }
  .btn {
    width: 100%; padding: 11px; background: #1a73e8; color: #fff;
    border: none; border-radius: 7px; font-size: 15px; font-weight: 600;
    cursor: pointer; transition: background .15s;
  }
  .btn:hover { background: #1558b0; }
  .error {
    background: #fde8e8; color: #c62828; border: 1px solid #f5c6cb;
    border-radius: 7px; padding: 10px 14px; font-size: 13px;
    margin-bottom: 18px;
  }
</style>
</head>
<body>
<div class="card">
  <div class="logo">
    <img src="{{ url_for('static', filename='app-logo.png') }}"
         alt="Telecom Voice-to-Text logo">
    <h1>SLT Call Center</h1>
    <p>Transcription Dashboard</p>
  </div>
  {% if error %}
  <div class="error">{{ error }}</div>
  {% endif %}
  <form method="POST" action="/login">
    <input type="hidden" name="next" value="{{ next }}">
    <div class="field">
      <label for="username">Username</label>
      <input type="text" id="username" name="username"
             autocomplete="username" autofocus required>
    </div>
    <div class="field">
      <label for="password">Password</label>
      <input type="password" id="password" name="password"
             autocomplete="current-password" required>
    </div>
    <button class="btn" type="submit">Sign In</button>
  </form>
</div>
</body>
</html>
"""

# ── Embedded HTML template ────────────────────────────────────────────────────
_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SLT Call Center — Transcription Dashboard</title>
<style>
  :root {
    --blue:   #1a73e8;
    --green:  #1e8e3e;
    --orange: #f29900;
    --red:    #d93025;
    --purple: #9c27b0;
    --bg:     #f8f9fa;
    --card:   #ffffff;
    --border: #e0e0e0;
    --text:   #202124;
    --muted:  #5f6368;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', Arial, sans-serif; background: var(--bg);
         color: var(--text); font-size: 14px; }
  header { background: var(--blue); color: #fff; padding: 14px 24px;
           display: flex; justify-content: space-between; align-items: center; gap: 16px; }
  .header-brand { display: flex; align-items: center; gap: 10px; min-width: 0; }
  .header-logo {
    width: 38px; height: 38px; object-fit: contain; flex: 0 0 auto;
    background: #fff; border-radius: 8px;
  }
  header h1 { font-size: 18px; font-weight: 600; }
  header span { font-size: 12px; opacity: .8; }
  .header-actions { display: flex; align-items: center; gap: 14px; flex-wrap: wrap;
                    justify-content: flex-end; }
  .date-picker { background: rgba(255,255,255,.15); border: 1px solid rgba(255,255,255,.4);
                 color: #fff; padding: 5px 10px; border-radius: 6px; font-size: 13px;
                 cursor: pointer; }
  .date-picker::-webkit-calendar-picker-indicator { filter: invert(1); }
  .grid, .dashboard-section { display: flex; flex-direction: column; gap: 16px;
                              padding: 20px; }
  .metric-group { display: flex; flex-direction: column; gap: 12px; }
  .metric-group + .metric-group { margin-top: 4px; }
  .metric-group-header { display: flex; align-items: center; gap: 12px;
                         min-height: 18px; }
  .metric-group-title { color: var(--muted); font-size: 11px; font-weight: 800;
                        letter-spacing: .8px; text-transform: uppercase;
                        white-space: nowrap; }
  .metric-group-line { flex: 1; border-top: 1px solid var(--border); }
  .metric-row { display: grid; gap: 16px; align-items: stretch; }
  .metric-row > .card { height: 100%; }
  .metric-row.four { grid-template-columns: repeat(4, minmax(0, 1fr)); }
  .metric-row.three { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  .metric-row.two { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .metric-row.monitoring { grid-template-columns: 1fr 1fr 1fr 2fr; }
  .card { background: var(--card); border: 1px solid var(--border);
          border-radius: 8px; padding: 18px; }
  .card.wide { width: 100%; }
  .card h3 { font-size: 11px; text-transform: uppercase; letter-spacing: .6px;
             color: var(--muted); margin-bottom: 8px; }
  .stat { font-size: 28px; font-weight: 700; line-height: 1.1; }
  .stat.blue   { color: var(--blue);   }
  .stat.green  { color: var(--green);  }
  .stat.orange { color: var(--orange); }
  .stat.red    { color: var(--red);    }
  .stat.purple { color: var(--purple); }
  .sub { font-size: 12px; color: var(--muted); margin-top: 4px; }
  .safety-panel { border-left: 4px solid var(--border); }
  .safety-panel.ok { border-left-color: var(--green); }
  .safety-panel.warning { border-left-color: var(--orange); }
  .safety-panel.blocked,
  .safety-panel.db_unavailable { border-left-color: var(--red); }
  .safety-panel.disabled { border-left-color: var(--border); }
  .safety-line { display: flex; gap: 16px; flex-wrap: wrap; align-items: center; }
  .safety-line strong { color: var(--text); }
  .safety-status { font-weight: 800; text-transform: uppercase; letter-spacing: .4px; }
  .safety-status.ok { color: var(--green); }
  .safety-status.warning { color: var(--orange); }
  .safety-status.blocked,
  .safety-status.db_unavailable { color: var(--red); }
  .safety-status.disabled { color: var(--muted); }
  .safety-notice { border: 1px solid #f3b4af; background: #fde8e8;
                   color: #9f1d16; border-radius: 8px; padding: 12px 14px;
                   font-weight: 700; }
  /* Language bar */
  .lang-bar { display: flex; height: 12px; border-radius: 6px; overflow: hidden;
              margin: 8px 0; }
  .lang-bar span { display: block; }
  .lang-si { background: #1a73e8; }
  .lang-en { background: #1e8e3e; }
  .lang-ta { background: #f29900; }
  .lang-legend { display: flex; gap: 16px; font-size: 12px; color: var(--muted); }
  .lang-legend i { display: inline-block; width: 10px; height: 10px;
                   border-radius: 2px; margin-right: 4px; }
  /* Table */
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { background: #f1f3f4; font-size: 11px; text-transform: uppercase;
       letter-spacing: .4px; padding: 8px 10px; text-align: left;
       color: var(--muted); border-bottom: 2px solid var(--border); }
  td { padding: 8px 10px; border-bottom: 1px solid var(--border); vertical-align: middle; }
  tr:hover td { background: #f8f9fa; }
  .badge { display: inline-block; padding: 2px 7px; border-radius: 12px;
           font-size: 11px; font-weight: 600; }
  .badge.rt    { background: #e8f0fe; color: var(--blue); }
  .badge.archived { background: #e6f4ea; color: var(--green); }
  .tag { display: inline-block; padding: 1px 5px; border-radius: 3px;
         font-size: 10px; margin-right: 2px; }
  .tag-si { background: #e8f0fe; color: var(--blue); }
  .tag-en { background: #e6f4ea; color: var(--green); }
  .tag-ta { background: #fef7e0; color: #7a5800; }
  .actions { display: flex; gap: 6px; flex-wrap: wrap; }
  .action-btn { display: inline-block; padding: 4px 8px; border: 1px solid var(--border);
                border-radius: 5px; background: #fff; color: var(--blue);
                text-decoration: none; font-size: 11px; font-weight: 600; }
  .action-btn:hover { border-color: var(--blue); background: #f8fbff; }
  /* Trend bars */
  .trend-card { display: flex; flex-direction: column; min-height: 170px; }
  .trend { display: flex; align-items: flex-end; gap: 4px; height: 100px; }
  .trend-bar { flex: 1; background: var(--blue); border-radius: 3px 3px 0 0;
               min-height: 2px; position: relative; }
  .trend-bar:hover::after { content: attr(data-tip); position: absolute;
    bottom: 105%; left: 50%; transform: translateX(-50%);
    background: #333; color: #fff; padding: 3px 6px; border-radius: 4px;
    font-size: 11px; white-space: nowrap; }
  .refresh-note { text-align: center; padding: 8px; font-size: 11px;
                  color: var(--muted); }
  /* Model filter tabs */
  .filter-bar { display: flex; gap: 8px; padding: 12px 20px 0;
                flex-wrap: wrap; max-width: 100%; }
  .filter-btn { padding: 6px 14px; border-radius: 20px; border: 1px solid var(--border);
                background: #fff; font-size: 12px; cursor: pointer; font-weight: 600;
                color: var(--muted); transition: all .15s; }
  .filter-btn:hover  { border-color: var(--blue); color: var(--blue); }
  .filter-btn.active { background: var(--blue); color: #fff; border-color: var(--blue); }
  /* Model breakdown table */
  .model-breakdown { display: flex; gap: 10px; flex-wrap: wrap; }
  .model-chip { flex: 1; min-width: 180px; background: #f8f9fa; border-radius: 8px;
                padding: 12px 14px; border: 1px solid var(--border); }
  .model-chip .name  { font-size: 12px; font-weight: 700; color: var(--blue);
                       font-family: monospace; margin-bottom: 6px; }
  .model-chip .nums  { font-size: 12px; color: var(--muted); line-height: 1.8; }
  /* Daily cost table */
  .csv-btn { display: inline-block; padding: 6px 14px; background: var(--green);
             color: #fff; border-radius: 6px; font-size: 12px; font-weight: 600;
             text-decoration: none; margin-left: 10px; }
  .csv-btn:hover { background: #166d35; }
  .range-controls { display: flex; gap: 10px; align-items: flex-end;
                    flex-wrap: wrap; margin: 12px 0; }
  .range-field { display: flex; flex-direction: column; gap: 4px; }
  .range-field label { font-size: 11px; text-transform: uppercase;
                       letter-spacing: .4px; color: var(--muted);
                       font-weight: 700; }
  .range-field input { border: 1px solid var(--border); border-radius: 6px;
                       padding: 6px 8px; font-size: 13px; }
  .range-btn { border: 1px solid var(--border); border-radius: 6px;
               background: #fff; color: var(--blue); font-size: 12px;
               font-weight: 700; padding: 7px 12px; cursor: pointer; }
  .range-btn.primary { background: var(--blue); border-color: var(--blue);
                       color: #fff; }
  .range-note { font-size: 12px; color: var(--muted); margin-top: 2px; }
  .summary-card .stat { font-size: 22px; }
  .range-summary-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
                        gap: 12px; margin: 12px 0 14px; }
  .range-summary-card { background: #f8f9fa; border: 1px solid var(--border);
                        border-radius: 8px; padding: 14px; }
  .range-summary-card h4 { font-size: 11px; text-transform: uppercase;
                           letter-spacing: .6px; color: var(--muted);
                           margin-bottom: 8px; }
  .range-summary-card .stat { font-size: 22px; }
  .range-dates { font-family: Consolas, 'Courier New', monospace; }
  .month-row td { background: #e8f0fe; color: var(--blue); font-weight: 700;
                  text-transform: uppercase; letter-spacing: .4px; }
  .subtotal-row td { background: #f1f3f4; font-weight: 700; }
  .grand-total-row td { background: #fff4e5; font-weight: 800;
                        border-top: 2px solid var(--orange); }
  @media (max-width: 1100px) {
    .metric-row.four,
    .metric-row.three,
    .metric-row.two,
    .metric-row.monitoring { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  }
  @media (max-width: 640px) {
    header { align-items: flex-start; flex-direction: column; padding: 12px 16px; }
    header h1 { font-size: 16px; }
    .header-logo { width: 34px; height: 34px; }
    .header-actions { justify-content: flex-start; gap: 10px; width: 100%; }
    .grid, .dashboard-section { padding: 14px; }
    .metric-row.four,
    .metric-row.three,
    .metric-row.two,
    .metric-row.monitoring,
    .range-summary-grid { grid-template-columns: 1fr; }
    .metric-group-header { gap: 8px; }
    .metric-group-title { white-space: normal; }
  }
</style>
</head>
<body>

<header>
  <div class="header-brand">
    <img class="header-logo" src="{{ url_for('static', filename='app-logo.png') }}"
         alt="Telecom Voice-to-Text logo">
    <h1>SLT Call Center - Transcription Dashboard</h1>
  </div>
  <div class="header-actions">
    <span id="clock"></span>
    <input type="date" id="date-picker" class="date-picker" onchange="setDate(this.value)">
    <a href="/logout" style="background:rgba(255,255,255,.18);border:1px solid rgba(255,255,255,.4);
       color:#fff;padding:5px 13px;border-radius:6px;font-size:12px;font-weight:600;
       text-decoration:none;">Sign Out</a>
  </div>
</header>

<div id="filter-bar" class="filter-bar"></div>
<div id="root"><p style="padding:20px">Loading…</p></div>

<script>
// Empty date lets the server choose today in APP_TIMEZONE.
let activeDate = '';
let activeModel = 'all';
let activeStartDate = '';
let activeEndDate = '';

async function load() {
  const params = new URLSearchParams({
    model: activeModel,
    date: activeDate,
    start_date: activeStartDate,
    end_date: activeEndDate
  });
  const r = await fetch(`/api/data?${params.toString()}`);
  const d = await r.json();
  render(d);
}

function setDate(val) {
  activeDate = val;
  counter = 10;
  load();
}

function applyCostRange() {
  activeStartDate = document.getElementById('range-start').value || '';
  activeEndDate = document.getElementById('range-end').value || '';
  counter = 10;
  load();
}

function resetCostRange() {
  activeStartDate = '';
  activeEndDate = '';
  counter = 10;
  load();
}

function fmt(n, dp=2) {
  return Number(n||0).toLocaleString('en-US', {minimumFractionDigits:dp, maximumFractionDigits:dp});
}
function fmtInt(n) {
  return Number(n||0).toLocaleString('en-US');
}

function langTags(str) {
  if (!str) return '<span style="color:#999">—</span>';
  return str.split(',').map(l => {
    l = l.trim();
    const cls = l==='Sinhala' ? 'tag-si' : l==='English' ? 'tag-en' : 'tag-ta';
    return `<span class="tag ${cls}">${l}</span>`;
  }).join('');
}

function render(d) {
  const t  = d.today  || {};
  const tt = d.totals || {};
  const mt = d.month  || {};
  const lk = d.languages || {};
  const cp = d.cost_period || {};
  const rt = d.range_total || {};
  const rc = d.rolling_costs || {};
  const pme = d.projected_month_end || {};
  const periodStart = cp.start_date || activeStartDate || '';
  const periodEnd = cp.end_date || activeEndDate || '';
  const periodRangeText = periodStart && periodEnd ? `${periodStart} to ${periodEnd}` : '';

  // Sync date picker to selected date
  const dp = document.getElementById('date-picker');
  if (dp) dp.value = d.selected_date || activeDate;

  const dateLabel = d.is_today ? 'Today' : d.selected_date;
  const safety = d.daily_cost_safety || {};
  const safetyStatus = safety.status || 'disabled';
  const safetyLabels = {
    disabled: 'Disabled',
    ok: 'OK',
    warning: 'Warning',
    blocked: 'Blocked',
    db_unavailable: 'DB unavailable'
  };
  const safetyLabel = safetyLabels[safetyStatus] || safetyStatus;
  const safetyEnabled = Boolean(safety.enabled);
  const safetyBlocked = Boolean(safety.blocked);
  const safetyNoticeText = safetyStatus === 'db_unavailable'
    ? 'Daily cost safety cannot check MySQL metadata. Because the failure policy is block, new audio files will be deferred and not sent to Gemini.'
    : 'Daily cost limit reached. New audio files will be deferred and not sent to Gemini.';
  const safetyNotice = safetyBlocked ? `
    <div class="safety-notice">
      ${safetyNoticeText}
    </div>` : '';
  const safetyDetails = safetyEnabled ? `
    <div class="safety-line">
      <span>Status: <span class="safety-status ${safetyStatus}">${safetyLabel}</span></span>
      <span>Limit: <strong>Rs.${fmt(safety.limit_lkr,2)}</strong></span>
      <span>Used today: <strong>Rs.${fmt(safety.used_lkr,2)}</strong></span>
      <span>Remaining: <strong>Rs.${fmt(safety.remaining_lkr,2)}</strong></span>
      <span>Usage: <strong>${fmt(safety.usage_percent,1)}%</strong></span>
    </div>
    <div class="sub">${safety.reason || ''}</div>
    ${safety.error ? `<div class="sub">Error: ${safety.error}</div>` : ''}` :
    `<div class="safety-line">
      <span>Daily cost limit: <span class="safety-status disabled">Disabled</span></span>
    </div>`;
  const safetyCard = `
    <div class="card wide safety-panel ${safetyStatus}">
      <h3>Daily Cost Safety</h3>
      ${safetyDetails}
    </div>`;

  const total_lang = (lk.Sinhala||0) + (lk.English||0) + (lk.Tamil||0) || 1;
  const pSi = ((lk.Sinhala||0)/total_lang*100).toFixed(1);
  const pEn = ((lk.English||0)/total_lang*100).toFixed(1);
  const pTa = ((lk.Tamil  ||0)/total_lang*100).toFixed(1);

  // Trend chart: backend keeps 90 days for history; this chart renders latest 14.
  const trendDaily = (d.daily||[]).slice(0, 14).reverse();
  const maxCost = Math.max(...trendDaily.map(x=>x.cost_lkr), 1);
  const bars = trendDaily.map(x => {
    const h = Math.max(3, Math.round((x.cost_lkr/maxCost)*94));
    return `<div class="trend-bar" style="height:${h}px"
      data-tip="${x.day}: Rs.${fmt(x.cost_lkr)} (${x.calls} calls)"></div>`;
  }).join('');

  // Recent calls table
  const rows = (d.recent||[]).map(c => {
    const hasTranscript = (Number(c.success) === 1) || Boolean(c.transcript_file_path);
    const transcriptActions = hasTranscript && c.id ? `
      <div class="actions">
        <a class="action-btn" href="/api/transcripts/${encodeURIComponent(c.id)}" target="_blank" rel="noopener">View</a>
        <a class="action-btn" href="/api/transcripts/${encodeURIComponent(c.id)}/download">Download TXT</a>
      </div>` : '<span style="color:#999">—</span>';
    return `
    <tr>
      <td>${c.filename}</td>
      <td>${fmt(c.duration_seconds,1)}s
        ${c.silence_removed_seconds>0 ? `<br><small style="color:#1e8e3e">-${fmt(c.silence_removed_seconds,1)}s silence</small>` : ''}
      </td>
      <td>${fmtInt(c.total_tokens)}</td>
      <td>$${fmt(c.total_cost_usd,5)}<br><small>Rs.${fmt(c.total_cost_lkr,3)}</small></td>
      <td>${langTags(c.languages_detected)}</td>
      <td><span class="badge ${c.batch_mode ? 'archived':'rt'}">${c.batch_mode ? 'Archived':'Realtime'}</span></td>
      <td>${transcriptActions}</td>
      <td style="color:#999;font-size:11px">${(c.processed_at||'').slice(0,16).replace('T',' ')}</td>
    </tr>`;
  }).join('');

  // ── Model filter tabs ─────────────────────────────────────────────
  const allModels = ['all', ...(d.all_models||[])];
  const filterBar = allModels.map(m => {
    const label = m === 'all' ? '📊 All Models' : m;
    const cls   = m === activeModel ? 'active' : '';
    return `<button class="filter-btn ${cls}" onclick="setModel('${m}')">${label}</button>`;
  }).join('');
  document.getElementById('filter-bar').innerHTML = filterBar;

  // ── Model breakdown chips ──────────────────────────────────────────
  const modelChips = (d.model_breakdown||[]).map(m => `
    <div class="model-chip">
      <div class="name">${m.model}</div>
      <div class="nums">
        📞 ${fmtInt(m.calls)} calls<br>
        💰 Rs.${fmt(m.cost_lkr,2)}<br>
        🔢 ${fmtInt(m.tokens)} tokens<br>
        📊 Rs.${fmt(m.cost_lkr/Math.max(m.calls,1),3)} / call
      </div>
    </div>`).join('') || '<span style="color:#999;font-size:13px">No data yet</span>';

  // Monthly cost table rows
  const monthlyRows = (d.monthly||[]).map(row => {
    const avgPerCall = row.calls > 0 ? (row.cost_lkr / row.calls) : 0;
    return `<tr>
      <td style="font-weight:600">${row.label || row.month}</td>
      <td>${fmtInt(row.calls)}</td>
      <td>${fmt((row.audio_seconds||0)/60, 1)}</td>
      <td>${fmtInt(row.tokens)}</td>
      <td style="color:var(--orange)">$${fmt(row.cost_usd, 5)}</td>
      <td style="color:var(--red);font-weight:600">Rs.${fmt(row.cost_lkr, 2)}</td>
      <td style="color:var(--muted)">Rs.${fmt(avgPerCall, 3)}</td>
    </tr>`;
  }).join('') || '<tr><td colspan="7" style="text-align:center;color:#999;padding:16px">No data yet</td></tr>';

  // Daily cost table rows grouped by backend-calculated calendar month totals.
  const dailyGroups = d.daily_by_month || [];
  const dailyRows = dailyGroups.map(group => {
    const dayRows = (group.rows || []).map(row => `
      <tr>
        <td style="font-weight:600">${row.day}</td>
        <td>${fmtInt(row.calls)}</td>
        <td>${fmt(row.audio_minutes, 1)}</td>
        <td>${fmtInt(row.tokens)}</td>
        <td style="color:var(--orange)">$${fmt(row.cost_usd, 5)}</td>
        <td style="color:var(--red);font-weight:600">Rs.${fmt(row.cost_lkr, 2)}</td>
        <td style="color:var(--muted)">Rs.${fmt(row.avg_cost_per_call_lkr, 3)}</td>
        <td style="color:var(--muted)">Rs.${fmt(row.avg_cost_per_audio_min_lkr, 3)}</td>
      </tr>`).join('');
    const subtotal = group.subtotal || {};
    return `
      <tr class="month-row"><td colspan="8">${group.label || group.month}</td></tr>
      ${dayRows}
      <tr class="subtotal-row">
        <td>${group.label || group.month} Total</td>
        <td>${fmtInt(subtotal.calls)}</td>
        <td>${fmt(subtotal.audio_minutes, 1)}</td>
        <td>${fmtInt(subtotal.tokens)}</td>
        <td style="color:var(--orange)">$${fmt(subtotal.cost_usd, 5)}</td>
        <td style="color:var(--red)">Rs.${fmt(subtotal.cost_lkr, 2)}</td>
        <td>Rs.${fmt(subtotal.avg_cost_per_call_lkr, 3)}</td>
        <td>Rs.${fmt(subtotal.avg_cost_per_audio_min_lkr, 3)}</td>
      </tr>`;
  }).join('');
  const grandTotalRow = dailyGroups.length ? `
    <tr class="grand-total-row">
      <td>Grand Total</td>
      <td>${fmtInt(rt.calls)}</td>
      <td>${fmt(rt.audio_minutes, 1)}</td>
      <td>${fmtInt(rt.tokens)}</td>
      <td style="color:var(--orange)">$${fmt(rt.cost_usd, 5)}</td>
      <td style="color:var(--red)">Rs.${fmt(rt.cost_lkr, 2)}</td>
      <td>Rs.${fmt(rt.avg_cost_per_call_lkr, 3)}</td>
      <td>Rs.${fmt(rt.avg_cost_per_audio_min_lkr, 3)}</td>
    </tr>` : '';
  const groupedDailyRows = dailyRows + grandTotalRow ||
    '<tr><td colspan="8" style="text-align:center;color:#999;padding:16px">No data yet</td></tr>';

  const dailyCsvHref = `/api/daily-cost.csv?model=${encodeURIComponent(activeModel)}&start_date=${encodeURIComponent(periodStart)}&end_date=${encodeURIComponent(periodEnd)}`;
  const monthlyCsvHref = `/api/monthly-cost.csv?model=${encodeURIComponent(activeModel)}`;

  document.getElementById('root').innerHTML = `
  <div class="dashboard-section">
    ${safetyNotice}
    ${safetyCard}

    <section class="metric-group">
      <div class="metric-group-header">
        <div class="metric-group-title">Today’s Operations</div>
        <div class="metric-group-line"></div>
      </div>
      <div class="metric-row four">
        <!-- Calls today -->
        <div class="card">
          <h3>Calls — ${dateLabel}</h3>
          <div class="stat blue">${fmtInt(t.calls_today)}</div>
          <div class="sub">
            ${fmtInt(t.realtime_calls||0)} realtime &nbsp;·&nbsp;
            ${fmtInt(t.batch_calls||0)} archived
          </div>
        </div>

        <!-- Estimated cost today USD -->
        <div class="card">
          <h3>Estimated Cost — ${dateLabel} (USD)</h3>
          <div class="stat orange">$${fmt(t.cost_usd,4)}</div>
          <div class="sub">Per call avg: $${fmt((t.cost_usd||0)/(t.calls_today||1),5)}</div>
        </div>

        <!-- Estimated cost today LKR -->
        <div class="card">
          <h3>Estimated Cost — ${dateLabel} (LKR)</h3>
          <div class="stat red">Rs.${fmt(t.cost_lkr,2)}</div>
          <div class="sub">Per call avg: Rs.${fmt((t.cost_lkr||0)/(t.calls_today||1),3)}</div>
        </div>

        <!-- Language breakdown -->
        <div class="card">
          <h3>Language Breakdown — ${dateLabel}</h3>
          <div class="lang-bar">
            <span class="lang-si" style="width:${pSi}%"></span>
            <span class="lang-en" style="width:${pEn}%"></span>
            <span class="lang-ta" style="width:${pTa}%"></span>
          </div>
          <div class="lang-legend">
            <span><i style="background:#1a73e8"></i>Sinhala ${pSi}%</span>
            <span><i style="background:#1e8e3e"></i>English ${pEn}%</span>
            <span><i style="background:#f29900"></i>Tamil ${pTa}%</span>
          </div>
        </div>
      </div>
    </section>

    <section class="metric-group">
      <div class="metric-group-header">
        <div class="metric-group-title">Usage and Processing Efficiency</div>
        <div class="metric-group-line"></div>
      </div>
      <div class="metric-row three">
        <!-- Tokens today -->
        <div class="card">
          <h3>Tokens — ${dateLabel}</h3>
          <div class="stat purple">${fmtInt(t.tokens_total)}</div>
          <div class="sub">
            Audio: ${fmtInt(t.tokens_audio)} &nbsp;·&nbsp;
            Output: ${fmtInt(t.tokens_output)}
          </div>
        </div>

        <!-- Silence savings -->
        <div class="card">
          <h3>Silence Stripped — ${dateLabel}</h3>
          <div class="stat green">${fmt((d.silence_saved_s||0)/60,1)} min</div>
          <div class="sub">Silence removed before Gemini submission</div>
        </div>

        <!-- Audio processed -->
        <div class="card">
          <h3>Audio Processed — ${dateLabel}</h3>
          <div class="stat blue">${fmt((t.audio_seconds||0)/60,1)} min</div>
          <div class="sub">${fmt(t.audio_seconds||0,0)}s total audio</div>
        </div>
      </div>
    </section>

    <section class="metric-group">
      <div class="metric-group-header">
        <div class="metric-group-title">Rolling Cost Monitoring</div>
        <div class="metric-group-line"></div>
      </div>
      <div class="metric-row monitoring">
        <div class="card summary-card">
          <h3>Rolling Last 7 Days</h3>
          <div class="stat blue">Rs.${fmt((rc.last_7_days||{}).cost_lkr,2)}</div>
          <div class="sub">${fmtInt((rc.last_7_days||{}).calls)} calls &nbsp;·&nbsp; $${fmt((rc.last_7_days||{}).cost_usd,4)}</div>
        </div>

        <div class="card summary-card">
          <h3>Rolling Last 30 Days</h3>
          <div class="stat blue">Rs.${fmt((rc.last_30_days||{}).cost_lkr,2)}</div>
          <div class="sub">${fmtInt((rc.last_30_days||{}).calls)} calls &nbsp;·&nbsp; $${fmt((rc.last_30_days||{}).cost_usd,4)}</div>
        </div>

        <div class="card summary-card">
          <h3>Rolling Last 90 Days</h3>
          <div class="stat blue">Rs.${fmt((rc.last_90_days||{}).cost_lkr,2)}</div>
          <div class="sub">${fmtInt((rc.last_90_days||{}).calls)} calls &nbsp;·&nbsp; $${fmt((rc.last_90_days||{}).cost_usd,4)}</div>
        </div>

        <!-- 14-day trend -->
        <div class="card trend-card">
          <h3>14-Day Cost Trend (LKR)</h3>
          <div class="trend">${bars || '<span style="color:#999;font-size:12px">No data yet</span>'}</div>
        </div>
      </div>
    </section>

    <section class="metric-group">
      <div class="metric-group-header">
        <div class="metric-group-title">Billing Summary</div>
        <div class="metric-group-line"></div>
      </div>
      <div class="metric-row two">
        <!-- Selected month total -->
        <div class="card">
          <h3>Estimated Cost This Month</h3>
          <div class="stat red">Rs.${fmt(mt.cost_lkr,2)}</div>
          <div class="sub">
            ${fmtInt(mt.calls||0)} calls &nbsp;·&nbsp;
            $${fmt(mt.cost_usd,4)} &nbsp;·&nbsp;
            ${mt.label || mt.month || 'Selected month'}
          </div>
        </div>

        <!-- All-time totals -->
        <div class="card">
          <h3>All-Time Estimated Totals</h3>
          <div class="stat green">${fmtInt(tt.total_calls)} calls</div>
          <div class="sub">
            $${fmt(tt.total_cost_usd,2)} &nbsp;·&nbsp;
            Rs.${fmt(tt.total_cost_lkr,2)} &nbsp;·&nbsp;
            ${fmtInt(tt.total_tokens)} tokens
          </div>
        </div>
      </div>
    </section>

    <!-- Model breakdown -->
    <div class="card wide">
      <h3>Estimated Gemini API Cost by Model — ${dateLabel} (click a tab above to filter everything)</h3>
      <div class="model-breakdown" style="margin-top:10px">${modelChips}</div>
    </div>

    <!-- Monthly cost history -->
    <div class="card wide">
      <h3 style="display:flex;align-items:center">
        Monthly Estimated Cost History (last 12 months)
        <a class="csv-btn" href="${monthlyCsvHref}" download>Download CSV</a>
      </h3>
      <table style="margin-top:10px">
        <thead>
          <tr>
            <th>Month</th>
            <th>Calls</th>
            <th>Audio (min)</th>
            <th>Tokens</th>
            <th>Estimated Cost (USD)</th>
            <th>Estimated Cost (LKR)</th>
            <th>Avg / Call (LKR)</th>
          </tr>
        </thead>
        <tbody id="monthly-tbody"></tbody>
      </table>
    </div>

    <!-- Daily cost history -->
    <div class="card wide">
      <h3 style="display:flex;align-items:center">
        Daily Estimated Cost History
        <a class="csv-btn" href="${dailyCsvHref}" download>Download CSV</a>
      </h3>
      <div class="range-controls">
        <div class="range-field">
          <label for="range-start">Start date</label>
          <input type="date" id="range-start" value="${periodStart}">
        </div>
        <div class="range-field">
          <label for="range-end">End date</label>
          <input type="date" id="range-end" value="${periodEnd}">
        </div>
        <button type="button" class="range-btn primary" onclick="applyCostRange()">Apply</button>
        <button type="button" class="range-btn" onclick="resetCostRange()">Reset to current month</button>
      </div>
      <div class="range-note">Showing estimated Gemini API usage cost for ${cp.label || 'the selected period'}.</div>
      <div class="range-summary-grid">
        <div class="range-summary-card">
          <h4>Estimated Cost — Selected Range</h4>
          <div class="stat red">Rs.${fmt(rt.cost_lkr,2)}</div>
          <div class="sub">$${fmt(rt.cost_usd,4)} &nbsp;·&nbsp; ${cp.label || 'Selected range'}</div>
          ${periodRangeText ? `<div class="sub range-dates">${periodRangeText}</div>` : ''}
        </div>
        <div class="range-summary-card">
          <h4>Projected Month-End Cost — ${cp.label || 'Selected month'}</h4>
          <div class="stat orange">Rs.${fmt(pme.projected_cost_lkr,2)}</div>
          <div class="sub">Month-to-date: Rs.${fmt(pme.month_to_date_cost_lkr,2)} &nbsp;·&nbsp; ${fmtInt(pme.elapsed_days)} of ${fmtInt(pme.days_in_month)} days elapsed</div>
        </div>
      </div>
      <table style="margin-top:10px">
        <thead>
          <tr>
            <th>Date</th>
            <th>Calls</th>
            <th>Audio (min)</th>
            <th>Tokens</th>
            <th>Estimated Cost (USD)</th>
            <th>Estimated Cost (LKR)</th>
            <th>Avg / Call (LKR)</th>
            <th>Avg / Audio Min (LKR)</th>
          </tr>
        </thead>
        <tbody id="daily-tbody"></tbody>
      </table>
    </div>

    <!-- Recent calls -->
    <div class="card wide">
      <h3>Calls on ${dateLabel} (last 20)</h3>
      <table>
        <thead>
          <tr>
            <th>File</th>
            <th>Duration</th>
            <th>Tokens</th>
            <th>Estimated Cost</th>
            <th>Languages</th>
            <th>Mode</th>
            <th>Transcript</th>
            <th>Time</th>
          </tr>
        </thead>
        <tbody>${rows || '<tr><td colspan="8" style="text-align:center;color:#999;padding:20px">No calls yet — drop audio files into input_audio/ to begin</td></tr>'}</tbody>
      </table>
    </div>

  </div>
  <p class="refresh-note">Auto-refreshes every 10 seconds &nbsp;·&nbsp; Next update in <span id="countdown">10</span>s</p>
  `;

  // Populate history tables after root is rendered
  const monthlyTbody = document.getElementById('monthly-tbody');
  if (monthlyTbody) monthlyTbody.innerHTML = monthlyRows;

  const dailyTbody = document.getElementById('daily-tbody');
  if (dailyTbody) dailyTbody.innerHTML = groupedDailyRows;
}

// Clock
function tick() {
  document.getElementById('clock').textContent =
    new Date().toLocaleString('en-GB', {hour12:false});
}
tick();
setInterval(tick, 1000);

// Load immediately, then auto-refresh every 10 seconds
load();
setInterval(load, 10000);

// Countdown timer
let counter = 10;
setInterval(() => {
  counter = counter <= 1 ? 10 : counter - 1;
  const el = document.getElementById('countdown');
  if (el) el.textContent = counter;
}, 1000);

function setModel(m) {
  activeModel = m;
  counter = 10;
  load();
}
</script>
</body>
</html>
"""


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user"):
        return redirect(url_for("index"))
    error = None
    next_url = request.args.get("next") or request.form.get("next") or "/"
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = _find_user(username)
        if user and check_password_hash(user["password"], password):
            session["user"] = user["username"]
            session["role"] = user.get("role", "viewer")
            return redirect(next_url)
        error = "Invalid username or password."
    return render_template_string(_LOGIN_HTML, error=error, next=next_url)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def index():
    return render_template_string(_HTML)


@app.route("/api/data")
@login_required
def api_data():
    model = request.args.get("model", "all")
    date  = request.args.get("date", "")
    start_date = request.args.get("start_date", "")
    end_date = request.args.get("end_date", "")
    return jsonify(
        get_dashboard_data(
            model_filter=model,
            date=date,
            start_date=start_date,
            end_date=end_date,
        )
    )


def _safe_transcript_file(stored_path: str) -> Path | None:
    if not stored_path:
        return None
    try:
        base_dir = TRANSCRIPT_OUTPUT_DIR.resolve()
        resolved = resolve_transcript_path(stored_path)
        resolved.relative_to(base_dir)
    except (OSError, RuntimeError, ValueError):
        return None
    return resolved if resolved.is_file() else None


def _record_or_404(call_id: int) -> dict:
    record = get_call_transcript(call_id)
    if record is None:
        abort(404)
    return record


def _transcript_text(record: dict) -> str:
    transcript_file = _safe_transcript_file(record.get("transcript_file_path", ""))
    if transcript_file is not None:
        return transcript_file.read_text(encoding="utf-8")
    return record.get("transcript", "") or ""


def _download_name(record: dict) -> str:
    stem = Path(record.get("filename") or f"call-{record.get('id', 'transcript')}").stem
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", stem).strip(".-_") or "transcript"
    return f"{stem}_transcript.txt"


@app.route("/api/transcripts/<int:call_id>")
@login_required
def api_transcript_view(call_id: int):
    record = _record_or_404(call_id)
    text = _transcript_text(record)
    if not text:
        abort(404)
    return Response(text, content_type="text/plain; charset=utf-8")


@app.route("/api/transcripts/<int:call_id>/download")
@login_required
def api_transcript_download(call_id: int):
    record = _record_or_404(call_id)
    transcript_file = _safe_transcript_file(record.get("transcript_file_path", ""))
    if transcript_file is not None:
        return send_file(
            transcript_file,
            as_attachment=True,
            download_name=transcript_file.name,
            mimetype="text/plain",
        )

    text = record.get("transcript", "") or ""
    if not text:
        abort(404)
    response = Response(text, content_type="text/plain; charset=utf-8")
    response.headers["Content-Disposition"] = (
        f'attachment; filename="{_download_name(record)}"'
    )
    return response


@app.route("/api/daily-cost.csv")
@login_required
def api_daily_cost_csv():
    model = request.args.get("model", "all")
    start_date = request.args.get("start_date", "")
    end_date = request.args.get("end_date", "")
    data = get_dashboard_data(
        model_filter=model,
        date="",
        start_date=start_date,
        end_date=end_date,
    )

    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow([
        "Row Type",
        "Month",
        "Date",
        "Calls",
        "Audio (min)",
        "Tokens",
        "Estimated Cost (USD)",
        "Estimated Cost (LKR)",
        "Avg / Call (LKR)",
        "Avg / Audio Min (LKR)",
    ])

    for group in data.get("daily_by_month", []):
        month_label = group.get("label") or group.get("month", "")
        for row in group.get("rows", []):
            writer.writerow([
                "day",
                month_label,
                row.get("day", ""),
                int(row.get("calls", 0) or 0),
                f"{float(row.get('audio_minutes', 0) or 0):.1f}",
                int(row.get("tokens", 0) or 0),
                f"{float(row.get('cost_usd', 0) or 0):.6f}",
                f"{float(row.get('cost_lkr', 0) or 0):.2f}",
                f"{float(row.get('avg_cost_per_call_lkr', 0) or 0):.3f}",
                f"{float(row.get('avg_cost_per_audio_min_lkr', 0) or 0):.3f}",
            ])

        subtotal = group.get("subtotal", {})
        writer.writerow([
            "month_subtotal",
            month_label,
            "",
            int(subtotal.get("calls", 0) or 0),
            f"{float(subtotal.get('audio_minutes', 0) or 0):.1f}",
            int(subtotal.get("tokens", 0) or 0),
            f"{float(subtotal.get('cost_usd', 0) or 0):.6f}",
            f"{float(subtotal.get('cost_lkr', 0) or 0):.2f}",
            f"{float(subtotal.get('avg_cost_per_call_lkr', 0) or 0):.3f}",
            f"{float(subtotal.get('avg_cost_per_audio_min_lkr', 0) or 0):.3f}",
        ])

    total = data.get("range_total", {})
    writer.writerow([
        "grand_total",
        data.get("cost_period", {}).get("label", ""),
        "",
        int(total.get("calls", 0) or 0),
        f"{float(total.get('audio_minutes', 0) or 0):.1f}",
        int(total.get("tokens", 0) or 0),
        f"{float(total.get('cost_usd', 0) or 0):.6f}",
        f"{float(total.get('cost_lkr', 0) or 0):.2f}",
        f"{float(total.get('avg_cost_per_call_lkr', 0) or 0):.3f}",
        f"{float(total.get('avg_cost_per_audio_min_lkr', 0) or 0):.3f}",
    ])

    csv_text = output.getvalue()
    response = Response(csv_text, content_type="text/csv; charset=utf-8")
    response.headers["Content-Disposition"] = "attachment; filename=slt_daily_cost.csv"
    return response


@app.route("/api/monthly-cost.csv")
@login_required
def api_monthly_cost_csv():
    model = request.args.get("model", "all")
    start_date = request.args.get("start_date", "")
    end_date = request.args.get("end_date", "")
    data = get_dashboard_data(
        model_filter=model,
        date="",
        start_date=start_date,
        end_date=end_date,
    )

    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow([
        "Month",
        "Calls",
        "Audio (min)",
        "Tokens",
        "Estimated Cost (USD)",
        "Estimated Cost (LKR)",
        "Avg per Call (LKR)",
    ])

    if start_date or end_date:
        monthly_rows = [
            {
                "label": group.get("label") or group.get("month", ""),
                **(group.get("subtotal", {}) or {}),
            }
            for group in data.get("daily_by_month", [])
        ]
    else:
        monthly_rows = data.get("monthly", [])

    for row in monthly_rows:
        calls = int(row.get("calls", 0) or 0)
        cost_lkr = float(row.get("cost_lkr", 0) or 0)
        avg = cost_lkr / calls if calls else 0
        audio_minutes = float(row.get("audio_minutes", 0) or 0)
        if not audio_minutes:
            audio_minutes = float(row.get("audio_seconds", 0) or 0) / 60
        writer.writerow([
            row.get("label") or row.get("month", ""),
            calls,
            f"{audio_minutes:.1f}",
            int(row.get("tokens", 0) or 0),
            f"{float(row.get('cost_usd', 0) or 0):.6f}",
            f"{cost_lkr:.2f}",
            f"{avg:.3f}",
        ])

    csv_text = output.getvalue()
    response = Response(csv_text, content_type="text/csv; charset=utf-8")
    response.headers["Content-Disposition"] = (
        "attachment; filename=slt_monthly_cost.csv"
    )
    return response


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optional local dashboard for realtime transcription metadata"
    )
    parser.add_argument("--port", "-p", type=int, default=5050)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1). Use a reverse proxy/access controls for remote access.",
    )
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print("  Telecom Voice-to-Text - Optional Local Dashboard")
    print(f"  Open in browser: http://{args.host}:{args.port}")
    if args.host not in {"127.0.0.1", "localhost"}:
        print("  WARNING: dashboard is not bound to localhost. Use access controls.")
    print(f"{'='*55}\n")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
