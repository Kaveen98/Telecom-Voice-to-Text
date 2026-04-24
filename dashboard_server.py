"""
dashboard_server.py
Flask web dashboard for the SLT Call Center transcription pipeline.

Usage:
    python dashboard_server.py              # opens on http://localhost:5050
    python dashboard_server.py --port 8080  # custom port

The dashboard auto-refreshes every 30 seconds and shows:
  - Calls transcribed today
  - Cost today (USD and LKR)
  - Tokens used (audio, text, output)
  - Silence stripped today (time saved + cost saved)
  - Language breakdown (Sinhala / English / Tamil)
  - Real-time vs batch split
  - 14-day cost trend
  - Last 20 calls with full detail

Requires:
    pip install flask --break-system-packages
"""
from __future__ import annotations

import argparse
import sys

try:
    from flask import Flask, jsonify, render_template_string, request
except ImportError:
    print("ERROR: Flask is not installed.")
    print("Run:  pip install flask --break-system-packages")
    sys.exit(1)

from config import DASHBOARD_HOST, DASHBOARD_PORT, ENABLE_RESET, LKR_RATE
from database import get_dashboard_data, reset_db

app = Flask(__name__)

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
           display: flex; justify-content: space-between; align-items: center; }
  header h1 { font-size: 18px; font-weight: 600; }
  header span { font-size: 12px; opacity: .8; }
  .reset-btn { background: rgba(255,255,255,.15); border: 1px solid rgba(255,255,255,.4);
               color: #fff; padding: 6px 14px; border-radius: 6px; font-size: 12px;
               font-weight: 600; cursor: pointer; transition: background .15s; }
  .reset-btn:hover { background: rgba(255,255,255,.28); }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px,1fr));
          gap: 16px; padding: 20px; }
  .card { background: var(--card); border: 1px solid var(--border);
          border-radius: 8px; padding: 18px; }
  .card.wide { grid-column: 1 / -1; }
  .card h3 { font-size: 11px; text-transform: uppercase; letter-spacing: .6px;
             color: var(--muted); margin-bottom: 8px; }
  .stat { font-size: 28px; font-weight: 700; line-height: 1.1; }
  .stat.blue   { color: var(--blue);   }
  .stat.green  { color: var(--green);  }
  .stat.orange { color: var(--orange); }
  .stat.red    { color: var(--red);    }
  .stat.purple { color: var(--purple); }
  .sub { font-size: 12px; color: var(--muted); margin-top: 4px; }
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
  .badge.batch { background: #e6f4ea; color: var(--green); }
  .tag { display: inline-block; padding: 1px 5px; border-radius: 3px;
         font-size: 10px; margin-right: 2px; }
  .tag-si { background: #e8f0fe; color: var(--blue); }
  .tag-en { background: #e6f4ea; color: var(--green); }
  .tag-ta { background: #fef7e0; color: #7a5800; }
  /* Trend bars */
  .trend { display: flex; align-items: flex-end; gap: 4px; height: 60px; }
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
</style>
</head>
<body>

<header>
  <h1>🎙 SLT Call Center - Transcription Dashboard</h1>
  <div style="display:flex;align-items:center;gap:14px">
    <span id="clock"></span>
    <button class="reset-btn" onclick="resetDashboard()">🗑 Reset Data</button>
  </div>
</header>

<div id="filter-bar" class="filter-bar"></div>
<div id="root"><p style="padding:20px">Loading…</p></div>

<script>
  const LKR_RATE = {{ lkr_rate }};

async function load() {
  const r = await fetch(`/api/data?model=${encodeURIComponent(activeModel)}`);
  const d = await r.json();
  render(d);
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
  const lk = d.languages || {};

  const total_lang = (lk.Sinhala||0) + (lk.English||0) + (lk.Tamil||0) || 1;
  const pSi = ((lk.Sinhala||0)/total_lang*100).toFixed(1);
  const pEn = ((lk.English||0)/total_lang*100).toFixed(1);
  const pTa = ((lk.Tamil  ||0)/total_lang*100).toFixed(1);

  // Silence savings in LKR: saved_seconds × 258 tokens/s / 1M × $1.00 audio rate × LKR
  const silSavedRs = ((d.silence_saved_s||0) * 258 / 1e6 * 1.0 * LKR_RATE).toFixed(2);

  // Trend chart
  const daily = (d.daily||[]).slice().reverse();
  const maxCost = Math.max(...daily.map(x=>x.cost_lkr), 1);
  const bars = daily.map(x => {
    const h = Math.max(2, Math.round((x.cost_lkr/maxCost)*56));
    return `<div class="trend-bar" style="height:${h}px"
      data-tip="${x.day}: Rs.${fmt(x.cost_lkr)} (${x.calls} calls)"></div>`;
  }).join('');

  // Recent calls table
  const rows = (d.recent||[]).map(c => `
    <tr>
      <td>${c.filename}</td>
      <td>${fmt(c.duration_seconds,1)}s
        ${c.silence_removed_seconds>0 ? `<br><small style="color:#1e8e3e">-${fmt(c.silence_removed_seconds,1)}s silence</small>` : ''}
      </td>
      <td>${fmtInt(c.total_tokens)}</td>
      <td>$${fmt(c.total_cost_usd,5)}<br><small>Rs.${fmt(c.total_cost_lkr,3)}</small></td>
      <td>${langTags(c.languages_detected)}</td>
      <td><span class="badge ${c.batch_mode ? 'batch':'rt'}">${c.batch_mode ? 'Batch':'Real-time'}</span></td>
      <td style="color:#999;font-size:11px">${(c.processed_at||'').slice(0,16).replace('T',' ')}</td>
    </tr>`).join('');

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

  document.getElementById('root').innerHTML = `
  <div class="grid">

    <!-- Calls today -->
    <div class="card">
      <h3>Calls Today</h3>
      <div class="stat blue">${fmtInt(t.calls_today)}</div>
      <div class="sub">
        ${fmtInt(t.realtime_calls||0)} real-time &nbsp;·&nbsp;
        ${fmtInt(t.batch_calls||0)} batch
      </div>
    </div>

    <!-- Cost today USD -->
    <div class="card">
      <h3>Cost Today (USD)</h3>
      <div class="stat orange">$${fmt(t.cost_usd,4)}</div>
      <div class="sub">Per call avg: $${fmt((t.cost_usd||0)/(t.calls_today||1),5)}</div>
    </div>

    <!-- Cost today LKR -->
    <div class="card">
      <h3>Cost Today (LKR)</h3>
      <div class="stat red">Rs.${fmt(t.cost_lkr,2)}</div>
      <div class="sub">Per call avg: Rs.${fmt((t.cost_lkr||0)/(t.calls_today||1),3)}</div>
    </div>

    <!-- Tokens today -->
    <div class="card">
      <h3>Tokens Today</h3>
      <div class="stat purple">${fmtInt(t.tokens_total)}</div>
      <div class="sub">
        Audio: ${fmtInt(t.tokens_audio)} &nbsp;·&nbsp;
        Output: ${fmtInt(t.tokens_output)}
      </div>
    </div>

    <!-- Silence savings -->
    <div class="card">
      <h3>Silence Stripped Today</h3>
      <div class="stat green">${fmt((d.silence_saved_s||0)/60,1)} min</div>
      <div class="sub">Saved ≈ Rs.${silSavedRs} in audio token cost</div>
    </div>

    <!-- Audio processed -->
    <div class="card">
      <h3>Audio Processed Today</h3>
      <div class="stat blue">${fmt((t.audio_seconds||0)/60,1)} min</div>
      <div class="sub">${fmt(t.audio_seconds||0,0)}s total audio</div>
    </div>

    <!-- Language breakdown -->
    <div class="card">
      <h3>Language Breakdown (Today)</h3>
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

    <!-- 14-day trend -->
    <div class="card">
      <h3>14-Day Cost Trend (LKR)</h3>
      <div class="trend">${bars || '<span style="color:#999;font-size:12px">No data yet</span>'}</div>
    </div>

    <!-- All-time totals -->
    <div class="card">
      <h3>All-Time Totals</h3>
      <div class="stat green">${fmtInt(tt.total_calls)} calls</div>
      <div class="sub">
        $${fmt(tt.total_cost_usd,2)} &nbsp;·&nbsp;
        Rs.${fmt(tt.total_cost_lkr,2)} &nbsp;·&nbsp;
        ${fmtInt(tt.total_tokens)} tokens
      </div>
    </div>

    <!-- Model breakdown -->
    <div class="card wide">
      <h3>Cost by Model — Today (click a tab above to filter everything)</h3>
      <div class="model-breakdown" style="margin-top:10px">${modelChips}</div>
    </div>

    <!-- Recent calls -->
    <div class="card wide">
      <h3>Recent Calls (last 20)</h3>
      <table>
        <thead>
          <tr>
            <th>File</th>
            <th>Duration</th>
            <th>Tokens</th>
            <th>Cost</th>
            <th>Languages</th>
            <th>Mode</th>
            <th>Time</th>
          </tr>
        </thead>
        <tbody>${rows || '<tr><td colspan="7" style="text-align:center;color:#999;padding:20px">No calls yet — drop audio files into input_audio/incoming/ to begin</td></tr>'}</tbody>
      </table>
    </div>

  </div>
  <p class="refresh-note">Auto-refreshes every 10 seconds &nbsp;·&nbsp; Rate: 1 USD = Rs.${LKR_RATE} &nbsp;·&nbsp; Next update in <span id="countdown">10</span>s</p>
  `;
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

// Reset all data
async function resetDashboard() {
  if (!confirm('Reset all call records from the database?\\nThis cannot be undone.')) return;
  await fetch('/api/reset', { method: 'POST' });
  activeModel = 'all';
  load();
}

// Active model filter
let activeModel = 'all';
function setModel(m) {
  activeModel = m;
  counter = 10;
  load();
}
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(_HTML, lkr_rate=LKR_RATE)


@app.route("/api/data")
def api_data():
    model = request.args.get("model", "all")
    return jsonify(get_dashboard_data(model_filter=model))


@app.route("/api/reset", methods=["POST"])
def api_reset():
    if not ENABLE_RESET:
        return jsonify({
            "status": "disabled",
            "message": "Reset endpoint is disabled. Set STT_ENABLE_RESET=true to enable it.",
        }), 403
    reset_db()
    return jsonify({"status": "ok", "message": "All records cleared"})


def main() -> None:
    parser = argparse.ArgumentParser(description="SLT Transcription Dashboard")
    parser.add_argument("--port", "-p", type=int, default=DASHBOARD_PORT)
    parser.add_argument("--host", default=DASHBOARD_HOST)
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print("  SLT Call Center — Transcription Dashboard")
    print(f"  Open in browser: http://localhost:{args.port}")
    print(f"{'='*55}\n")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
