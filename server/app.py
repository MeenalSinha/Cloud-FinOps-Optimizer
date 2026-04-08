"""
FastAPI server for the Cloud FinOps Optimizer.

Uses create_fastapi_app() from openenv.core.env_server which automatically
provides: /ws  /step  /state  /health  /web  /docs

Custom endpoints defined here (overriding or supplementing the framework):
    POST /reset     -- explicit override guaranteeing task_id body routing
    GET  /grade     -- deterministic episode score (0.0-1.0) + explainability
    GET  /tasks     -- list available tasks
    POST /simulate  -- project action outcome without mutating state

/reset override rationale:
    create_fastapi_app() may or may not forward JSON body kwargs to env.reset().
    Rather than rely on framework internals, we define /reset explicitly here so
    {"task_id": "task1"} in the request body is always forwarded correctly.
    The framework's WebSocket /ws endpoint uses its own session management for
    interactive clients; our HTTP /reset serves the validator and inference script.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse
from openenv.core.env_server import create_fastapi_app
from pydantic import BaseModel

from models import FinOpsAction, FinOpsObservation
from server.environment import CloudFinOpsEnvironment, TASK_CONFIGS

# ---------------------------------------------------------------------------
# Module-level environment instance.
# Shared by all HTTP endpoints. The WebSocket path managed by create_fastapi_app
# uses its own instance; we keep them in sync via the startup hook + middleware.
# ---------------------------------------------------------------------------
_env = CloudFinOpsEnvironment()

# ---------------------------------------------------------------------------
# Build the base OpenEnv application
# ---------------------------------------------------------------------------
def _env_factory():
    return _env

app = create_fastapi_app(_env_factory, FinOpsAction, FinOpsObservation)

# Remove the framework's /reset and /state so our explicit overrides take effect
app.router.routes = [
    r for r in app.router.routes 
    if getattr(r, "path", None) not in ("/reset", "/state")
]


def _get_env() -> CloudFinOpsEnvironment:
    """
    Return the authoritative environment instance.
    Prefers app.state.env (set by create_fastapi_app) if available,
    otherwise falls back to the module-level _env.
    """
    try:
        env = app.state.env
        if env is not None:
            return env
    except AttributeError:
        pass
    return _env


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to the interactive Web UI."""
    return RedirectResponse(url="/web")


# ---------------------------------------------------------------------------
# Startup hook
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def _startup():
    """Wire module-level _env into app.state if the framework has not done so."""
    try:
        if app.state.env is None:
            app.state.env = _env
    except AttributeError:
        app.state.env = _env


# ---------------------------------------------------------------------------
# Middleware: keep _env in sync after /step calls
# ---------------------------------------------------------------------------

@app.middleware("http")
async def _sync_env_middleware(request: Request, call_next):
    global _env
    response = await call_next(request)
    if request.url.path in ("/step",):
        try:
            fw_env = app.state.env
            if fw_env is not None and fw_env is not _env:
                _env = fw_env
        except AttributeError:
            pass
    return response


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "task1"


# ---------------------------------------------------------------------------
# /reset -- explicit override so task_id is always forwarded from body
# ---------------------------------------------------------------------------

@app.post("/reset", tags=["env"])
async def reset_episode(request_body: ResetRequest):
    """
    Start a new episode for the specified task.

    Body:
        {"task_id": "task1"}   easy   -- Waste Cleanup
        {"task_id": "task2"}   medium -- Resource Optimization
        {"task_id": "task3"}   hard   -- Strategic Planning

    Returns a FinOpsObservation with full resource list, dependency graph,
    SLA status, and the episode goal.
    """
    env = _get_env()
    try:
        obs = env.reset(task_id=request_body.task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Serialize -- FinOpsObservation is a Pydantic model
    try:
        return JSONResponse(content=obs.model_dump())
    except AttributeError:
        # Pydantic v1 fallback
        return JSONResponse(content=obs.dict())


# ---------------------------------------------------------------------------
# /grade -- deterministic score
# ---------------------------------------------------------------------------

@app.get("/grade", tags=["grading"])
async def grade_episode():
    """
    Return the deterministic episode grade (0.0-1.0) plus explainability_score.
    Safe to call at any point during or after an episode.
    """
    env = _get_env()
    if not env.state.task_id:
        raise HTTPException(
            status_code=400,
            detail="No active episode. Call /reset first.",
        )
    return JSONResponse(content=env.grade())


# ---------------------------------------------------------------------------
# /tasks -- task catalogue
# ---------------------------------------------------------------------------

@app.get("/tasks", tags=["meta"])
async def list_tasks():
    """List all available tasks with descriptions, budgets, and step limits."""
    return {
        task_id: {
            "description":     cfg["description"],
            "budget_per_hour": cfg["budget_per_hour"],
            "max_steps":       cfg["max_steps"],
        }
        for task_id, cfg in TASK_CONFIGS.items()
    }


# ---------------------------------------------------------------------------
# /simulate -- what-if projection
# ---------------------------------------------------------------------------

@app.post("/simulate", tags=["tools"])
async def simulate_action(request: Request):
    """
    Project the outcome of a proposed action without mutating environment state.

    Body (same format as a /step action):
        {"action_type": "terminate", "resource_id": "ebs-001"}
        {"action_type": "resize",    "resource_id": "ec2-t01", "target_size": "t2.medium"}
        {"action_type": "reserve",   "resource_id": "ec2-h01"}

    Returns SimulateResult:
        projected_cost_per_hour, projected_budget_remaining, projected_reward,
        projected_sla_violations, cascading_risks, recommendation, safe_to_apply
    """
    env = _get_env()
    if not env.state.task_id:
        raise HTTPException(
            status_code=400,
            detail="No active episode. Call /reset first.",
        )
    body = await request.json()
    result = env._simulate(body)
    try:
        return JSONResponse(content=result.model_dump())
    except AttributeError:
        return JSONResponse(content=result.dict())


# ---------------------------------------------------------------------------
# /state - explicit override to preserve custom subclass fields
# ---------------------------------------------------------------------------

@app.get("/state", tags=["State Management"])
async def get_state_override():
    """Return the raw FinOpsState so extended fields like task_id aren't dropped."""
    env = _get_env()
    try:
        return JSONResponse(content=env.state.model_dump())
    except AttributeError:
        return JSONResponse(content=env.state.dict())


# ---------------------------------------------------------------------------
# /web -- Interactive Demo UI
# ---------------------------------------------------------------------------

_WEB_UI = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cloud FinOps Optimizer</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #f5f6fa;
    --surface: #ffffff;
    --border: #e2e6ea;
    --border-light: #f0f2f5;
    --text: #1a1d23;
    --muted: #6b7280;
    --accent: #2563eb;
    --accent-light: #eff6ff;
    --green: #16a34a;
    --green-bg: #f0fdf4;
    --red: #dc2626;
    --red-bg: #fef2f2;
    --yellow: #d97706;
    --yellow-bg: #fffbeb;
    --purple: #7c3aed;
    --purple-bg: #f5f3ff;
    --orange: #ea580c;
    --shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04);
    --shadow-md: 0 4px 6px rgba(0,0,0,0.07), 0 2px 4px rgba(0,0,0,0.05);
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    font-size: 14px;
    min-height: 100vh;
  }
  header {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 0 24px;
    display: flex;
    align-items: center;
    gap: 14px;
    height: 56px;
    box-shadow: var(--shadow);
    position: sticky;
    top: 0;
    z-index: 10;
  }
  .logo {
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .logo-icon {
    width: 32px; height: 32px;
    background: var(--accent);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
  }
  .logo-icon svg { width: 18px; height: 18px; }
  header h1 { font-size: 16px; font-weight: 700; color: var(--text); letter-spacing: -0.3px; }
  .badge {
    background: var(--accent-light);
    color: var(--accent);
    border: 1px solid #bfdbfe;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.2px;
  }
  #episodeStatus {
    font-size: 12px;
    color: var(--muted);
    margin-left: auto;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 4px 10px;
    font-weight: 500;
  }
  .controls {
    display: flex;
    gap: 8px;
    padding: 12px 24px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    flex-wrap: wrap;
    align-items: center;
  }
  .btn {
    padding: 7px 16px;
    border-radius: 6px;
    border: 1px solid transparent;
    cursor: pointer;
    font-size: 13px;
    font-weight: 600;
    font-family: inherit;
    transition: all 0.15s;
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }
  .btn:hover { filter: brightness(0.93); transform: translateY(-1px); box-shadow: var(--shadow-md); }
  .btn:active { transform: translateY(0); filter: brightness(0.88); }
  .btn-primary { background: var(--accent); color: #fff; border-color: var(--accent); }
  .btn-green { background: var(--green); color: #fff; border-color: var(--green); }
  .btn-purple { background: var(--purple); color: #fff; border-color: var(--purple); }
  .btn-yellow { background: var(--yellow); color: #fff; border-color: var(--yellow); }
  .btn-outline { background: transparent; color: var(--text); border-color: var(--border); }
  .btn-outline:hover { background: var(--bg); }
  select.task-select {
    background: var(--surface);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 7px 12px;
    font-size: 13px;
    font-family: inherit;
    cursor: pointer;
    font-weight: 500;
    outline: none;
  }
  select.task-select:focus { border-color: var(--accent); box-shadow: 0 0 0 3px #dbeafe; }
  .layout {
    display: grid;
    grid-template-columns: 1fr 360px;
    gap: 16px;
    padding: 16px 24px;
    max-width: 1400px;
    margin: 0 auto;
  }
  .col-left, .col-right { display: flex; flex-direction: column; gap: 16px; }
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
    box-shadow: var(--shadow);
  }
  .card-header {
    padding: 12px 18px;
    border-bottom: 1px solid var(--border-light);
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--surface);
  }
  .card-header h2 {
    font-size: 12px;
    font-weight: 700;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.6px;
  }
  .card-body { padding: 16px 18px; }
  .status-pill {
    display: inline-flex;
    align-items: center;
    padding: 2px 9px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.2px;
  }
  .pill-green { background: var(--green-bg); color: var(--green); }
  .pill-red { background: var(--red-bg); color: var(--red); }
  .pill-yellow { background: var(--yellow-bg); color: var(--yellow); }
  .pill-gray { background: #f3f4f6; color: var(--muted); }
  .pill-purple { background: var(--purple-bg); color: var(--purple); }
  .pill-blue { background: var(--accent-light); color: var(--accent); }
  table { width: 100%; border-collapse: collapse; }
  th {
    text-align: left;
    font-size: 11px;
    font-weight: 700;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding: 8px 14px;
    border-bottom: 1px solid var(--border);
    background: #f8f9fb;
    white-space: nowrap;
  }
  td {
    padding: 9px 14px;
    border-bottom: 1px solid var(--border-light);
    font-size: 13px;
    vertical-align: middle;
  }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: #f8faff; }
  tr.sla-violation td { background: #fff5f5; }
  tr.terminated td { opacity: 0.45; }
  .cost-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .cost-card {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 16px;
  }
  .cost-card .label { font-size: 11px; color: var(--muted); font-weight: 600; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.3px; }
  .cost-card .value { font-size: 22px; font-weight: 700; letter-spacing: -0.5px; color: var(--text); }
  .cost-card .value.green { color: var(--green); }
  .cost-card .value.red { color: var(--red); }
  .cost-card .value.blue { color: var(--accent); }
  .cost-card .value.yellow { color: var(--yellow); }
  #costChart { max-height: 130px; }
  #depSvg { width: 100%; height: 240px; }
  .node circle {
    fill: white;
    stroke: var(--accent);
    stroke-width: 2;
    cursor: pointer;
    filter: drop-shadow(0 1px 2px rgba(0,0,0,0.08));
  }
  .node circle.critical { stroke: var(--red); fill: var(--red-bg); }
  .node circle.risk { stroke: var(--yellow); fill: var(--yellow-bg); }
  .node circle.sla { stroke: var(--orange); fill: #fff7ed; }
  .node text { font-size: 8.5px; fill: var(--text); text-anchor: middle; pointer-events: none; font-family: 'Inter', sans-serif; font-weight: 600; }
  .link { stroke: #d1d5db; stroke-width: 1.5; fill: none; }
  .link.risk { stroke: var(--yellow); stroke-width: 2; stroke-dasharray: 4 2; }
  #logList { list-style: none; max-height: 200px; overflow-y: auto; }
  #logList li { padding: 7px 0; border-bottom: 1px solid var(--border-light); font-size: 12.5px; line-height: 1.5; }
  #logList li:last-child { border-bottom: none; }
  .log-reward-pos { color: var(--green); font-weight: 700; }
  .log-reward-neg { color: var(--red); font-weight: 700; }
  .log-error { color: var(--yellow); font-weight: 600; }
  .log-time { color: var(--muted); font-size: 11px; margin-right: 6px; font-family: monospace; }
  .log-action { font-weight: 600; color: var(--text); }
  #simBox {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px;
  }
  .sim-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid var(--border-light);
    font-size: 12.5px;
  }
  .sim-row:last-of-type { border-bottom: none; }
  .sim-label { color: var(--muted); font-weight: 500; }
  .safe-yes { color: var(--green); font-weight: 700; }
  .safe-no { color: var(--red); font-weight: 700; }
  .spinner {
    width: 16px; height: 16px;
    border: 2px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    display: inline-block;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .empty-msg { text-align: center; padding: 32px; color: var(--muted); font-size: 13px; }
  .bar-wrap { display: inline-flex; align-items: center; gap: 6px; }
  .bar { height: 5px; border-radius: 3px; background: var(--accent); display: inline-block; }
  .bar.high { background: var(--red); }
  .bar.med { background: var(--yellow); }
  .bar.low { background: var(--green); }
  .divider { height: 1px; background: var(--border-light); margin: 10px 0; }
  .sim-note { font-size: 12px; color: var(--muted); line-height: 1.6; margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--border-light); }
  @media (max-width: 960px) { .layout { grid-template-columns: 1fr; padding: 12px; } }
  ::-webkit-scrollbar { width: 5px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
  .sep { width: 1px; height: 20px; background: var(--border); margin: 0 4px; }
</style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-icon">
      <svg viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M9 2L15 7V16H3V7L9 2Z" stroke="white" stroke-width="1.5" stroke-linejoin="round"/>
        <path d="M6 16V10H12V16" stroke="white" stroke-width="1.5" stroke-linejoin="round"/>
      </svg>
    </div>
    <h1>Cloud FinOps Optimizer</h1>
  </div>
  <span class="badge">OpenEnv</span>
  <div class="sep"></div>
  <span id="episodeStatus">No active episode</span>
</header>

<div class="controls">
  <select class="task-select" id="taskSelect">
    <option value="task1">Task 1 -- Waste Cleanup (Easy)</option>
    <option value="task2">Task 2 -- Resource Optimization (Medium)</option>
    <option value="task3">Task 3 -- Strategic Planning (Hard)</option>
  </select>
  <div class="sep"></div>
  <button class="btn btn-primary" onclick="doReset()">Reset Environment</button>
  <button class="btn btn-green" onclick="doStep()">Run Agent Step</button>
  <button class="btn btn-yellow" onclick="doSimulate()">Simulate Next</button>
  <button class="btn btn-purple" onclick="doGrade()">Get Score</button>
  <span id="loadingSpinner" style="display:none; margin-left:4px"><span class="spinner"></span></span>
</div>

<div class="layout">
  <div class="col-left">

    <!-- Cost Dashboard -->
    <div class="card">
      <div class="card-header"><h2>Cost Dashboard</h2></div>
      <div class="card-body">
        <div class="cost-grid">
          <div class="cost-card"><div class="label">Current Cost / hr</div><div class="value blue" id="metCurrent">--</div></div>
          <div class="cost-card"><div class="label">Initial Cost / hr</div><div class="value" id="metInitial">--</div></div>
          <div class="cost-card"><div class="label">Budget / hr</div><div class="value yellow" id="metBudget">--</div></div>
          <div class="cost-card"><div class="label">Cost Saved</div><div class="value green" id="metSaved">--</div></div>
        </div>
        <canvas id="costChart" style="margin-top:16px"></canvas>
      </div>
    </div>

    <!-- Resource Table -->
    <div class="card">
      <div class="card-header">
        <h2>Resources</h2>
        <span id="resCount" style="font-size:12px;color:var(--muted);font-weight:500"></span>
      </div>
      <div style="overflow-x:auto">
        <table id="resTable">
          <thead><tr>
            <th>Resource ID</th><th>Type</th><th>CPU %</th><th>Memory %</th>
            <th>Cost / hr</th><th>Status</th><th>Reserved</th><th>SLA</th><th>Idle</th>
          </tr></thead>
          <tbody id="resBody"><tr><td colspan="9" class="empty-msg">Reset the environment to load resources</td></tr></tbody>
        </table>
      </div>
    </div>

  </div>
  <div class="col-right">

    <!-- Dependency Graph -->
    <div class="card">
      <div class="card-header"><h2>Dependency Graph</h2></div>
      <div class="card-body" style="padding:10px">
        <svg id="depSvg">
          <defs>
            <marker id="arr" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
              <polygon points="0 0, 7 3.5, 0 7" fill="#9ca3af"/>
            </marker>
          </defs>
          <text x="50%" y="50%" text-anchor="middle" fill="#9ca3af" font-size="13" font-family="Inter,sans-serif">Reset to load graph</text>
        </svg>
        <div style="display:flex;gap:12px;margin-top:8px;flex-wrap:wrap">
          <span style="font-size:11px;color:var(--muted);display:flex;align-items:center;gap:4px"><span style="width:10px;height:10px;border-radius:50%;border:2px solid var(--accent);display:inline-block"></span>Normal</span>
          <span style="font-size:11px;color:var(--muted);display:flex;align-items:center;gap:4px"><span style="width:10px;height:10px;border-radius:50%;border:2px solid var(--red);background:var(--red-bg);display:inline-block"></span>Critical</span>
          <span style="font-size:11px;color:var(--muted);display:flex;align-items:center;gap:4px"><span style="width:10px;height:10px;border-radius:50%;border:2px solid var(--yellow);background:var(--yellow-bg);display:inline-block"></span>Risk</span>
        </div>
      </div>
    </div>

    <!-- Simulation Preview -->
    <div class="card">
      <div class="card-header"><h2>Simulation Preview</h2></div>
      <div class="card-body" style="padding:14px 18px">
        <div id="simBox"><p class="empty-msg" style="padding:16px 0">Click "Simulate Next" to preview projected outcome before committing an action.</p></div>
      </div>
    </div>

    <!-- Action Log -->
    <div class="card">
      <div class="card-header"><h2>Action Log</h2></div>
      <div class="card-body" style="padding:10px 18px">
        <ul id="logList"><li style="color:var(--muted);text-align:center;padding:16px 0">No actions yet</li></ul>
      </div>
    </div>

  </div>
</div>

<script>
// --- State -----------------------------------------------------------------
let costChart = null;
let stepCount = 0;
let initialCost = null;
let cascadeRisks = {};

// --- Chart -----------------------------------------------------------------
function initChart() {
  const ctx = document.getElementById('costChart').getContext('2d');
  if (costChart) costChart.destroy();
  costChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Cost/hr ($)',
        data: [],
        borderColor: '#2563eb',
        backgroundColor: 'rgba(37,99,235,0.07)',
        tension: 0.35,
        fill: true,
        pointRadius: 3,
        pointBackgroundColor: '#2563eb',
        borderWidth: 2,
      }]
    },
    options: {
      animation: { duration: 300 },
      plugins: { legend: { display: false } },
      scales: {
        x: { display: false },
        y: {
          ticks: { color: '#6b7280', font: { size: 10, family: 'Inter' } },
          grid: { color: '#f0f2f5' },
          border: { dash: [3, 3] }
        }
      }
    }
  });
}

function updateChart(cost) {
  if (!costChart) initChart();
  costChart.data.labels.push('S' + stepCount);
  costChart.data.datasets[0].data.push(parseFloat(cost.toFixed(4)));
  if (costChart.data.labels.length > 20) {
    costChart.data.labels.shift();
    costChart.data.datasets[0].data.shift();
  }
  costChart.update();
}

// --- Helpers ---------------------------------------------------------------
const $ = id => document.getElementById(id);
const show = id => $(id).style.display = '';
const hide = id => $(id).style.display = 'none';
function loading(on) { on ? show('loadingSpinner') : hide('loadingSpinner'); }

function setStatus(text, color) {
  const el = $('episodeStatus');
  el.textContent = text;
  el.style.color = color || 'var(--muted)';
}

function bar(pct, w = 54) {
  const cls = pct > 80 ? 'high' : pct > 55 ? 'med' : 'low';
  const px = Math.max(2, Math.round((pct / 100) * w));
  return `<span class="bar-wrap"><span class="bar ${cls}" style="width:${px}px"></span><span style="color:var(--muted);font-size:12px">${pct.toFixed(0)}%</span></span>`;
}

function statusPill(s) {
  const m = { running:'pill-green', terminated:'pill-gray', degraded:'pill-red', idle:'pill-yellow' };
  return `<span class="status-pill ${m[s]||'pill-gray'}">${s}</span>`;
}

function slaPill(s) {
  const m = { ok:'pill-green', at_risk:'pill-yellow', violated:'pill-red' };
  return `<span class="status-pill ${m[s]||'pill-gray'}">${s.replace('_',' ')}</span>`;
}

function timeStr() {
  return new Date().toLocaleTimeString('en-US', { hour12: false, hour:'2-digit', minute:'2-digit', second:'2-digit' });
}

// --- Resource Table --------------------------------------------------------
function renderResources(resources, slaViolations) {
  const slaSet = new Set(slaViolations || []);
  const tbody = $('resBody');
  $('resCount').textContent = resources.length + ' resources';
  tbody.innerHTML = '';
  resources.forEach(r => {
    const tr = document.createElement('tr');
    if (slaSet.has(r.id)) tr.classList.add('sla-violation');
    if (r.status === 'terminated') tr.classList.add('terminated');
    tr.innerHTML = `
      <td><span style="font-weight:600;color:var(--text)">${r.id}</span>${r.critical ? ' <span class="status-pill pill-red" style="font-size:10px;padding:1px 5px">critical</span>' : ''}</td>
      <td><span style="color:var(--muted);font-weight:500">${r.type.toUpperCase()}</span></td>
      <td>${bar(r.cpu_utilization)}</td>
      <td>${bar(r.memory_utilization)}</td>
      <td style="font-weight:600;color:var(--accent)">$${r.cost_per_hour.toFixed(4)}</td>
      <td>${statusPill(r.status)}</td>
      <td>${r.reserved ? '<span class="status-pill pill-purple">Yes</span>' : '<span style="color:var(--muted)">No</span>'}</td>
      <td>${slaPill(r.sla_status)}</td>
      <td style="color:${r.idle_hours>2?'var(--yellow)':'var(--muted)'};font-weight:${r.idle_hours>2?'600':'400'}">${r.idle_hours}h</td>
    `;
    tbody.appendChild(tr);
  });
}

// --- Cost Cards ------------------------------------------------------------
function fmt(v, d=4) { const n = parseFloat(v); return isNaN(n) ? '0.0000' : n.toFixed(d); }

function renderCost(obs, state) {
  const curr = parseFloat(obs.total_cost_per_hour) || 0;
  const budget = parseFloat(obs.budget_per_hour) || 0;
  if (initialCost === null) initialCost = curr;
  const saved = initialCost - curr;
  $('metCurrent').textContent = '$' + fmt(curr);
  const initVal = (state && state.initial_cost_per_hour) ? state.initial_cost_per_hour : initialCost;
  $('metInitial').textContent = '$' + fmt(initVal);
  $('metBudget').textContent = '$' + fmt(budget);
  $('metSaved').textContent = (saved >= 0 ? '+' : '') + '$' + fmt(saved);
  $('metSaved').className = 'value ' + (saved >= 0 ? 'green' : 'red');
  updateChart(curr);
}

// --- Dependency Graph (SVG) ------------------------------------------------
function renderDepGraph(resources, depGraph, cascRisks, slaViolations) {
  const svg = $('depSvg');
  svg.innerHTML = `<defs><marker id="arr" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><polygon points="0 0, 7 3.5, 0 7" fill="#9ca3af"/></marker></defs>`;
  const W = svg.clientWidth || 320, H = 240;
  const nodes = resources.filter(r => r.status !== 'terminated').map(r => r.id);
  const n = nodes.length;
  if (n === 0) {
    svg.innerHTML += '<text x="50%" y="50%" text-anchor="middle" fill="#9ca3af" font-size="13" font-family="Inter,sans-serif">No active resources</text>';
    return;
  }
  const riskSet = new Set(Object.keys(cascRisks || {}));
  const slaSet = new Set(slaViolations || []);
  const resMap = {}; resources.forEach(r => resMap[r.id] = r);
  const cx = W / 2, cy = H / 2, radius = Math.min(cx, cy) - 36;
  const pos = {};
  nodes.forEach((id, i) => {
    const angle = (2 * Math.PI * i / n) - Math.PI / 2;
    pos[id] = { x: cx + radius * Math.cos(angle), y: cy + radius * Math.sin(angle) };
  });
  // Edges
  (depGraph ? Object.entries(depGraph) : []).forEach(([src, targets]) => {
    (targets || []).forEach(tgt => {
      if (!pos[src] || !pos[tgt]) return;
      const isRisk = riskSet.has(src) || riskSet.has(tgt);
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', pos[src].x); line.setAttribute('y1', pos[src].y);
      line.setAttribute('x2', pos[tgt].x); line.setAttribute('y2', pos[tgt].y);
      line.setAttribute('class', 'link' + (isRisk ? ' risk' : ''));
      line.setAttribute('marker-end', 'url(#arr)');
      svg.appendChild(line);
    });
  });
  // Nodes
  nodes.forEach(id => {
    const p = pos[id]; const r = resMap[id];
    const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    g.setAttribute('class', 'node');
    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    circle.setAttribute('cx', p.x); circle.setAttribute('cy', p.y); circle.setAttribute('r', 20);
    let cls = '';
    if (r && r.critical) cls = 'critical';
    else if (riskSet.has(id)) cls = 'risk';
    else if (slaSet.has(id)) cls = 'sla';
    if (cls) circle.setAttribute('class', cls);
    const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    label.setAttribute('x', p.x); label.setAttribute('y', p.y);
    label.setAttribute('dy', '4'); label.setAttribute('font-family', 'Inter,sans-serif');
    label.textContent = id.slice(0, 7);
    g.appendChild(circle); g.appendChild(label);
    svg.appendChild(g);
  });
}

// --- Action Log ------------------------------------------------------------
function addLog(action, reward, error, slaViolations) {
  const ul = $('logList');
  if (ul.children[0] && ul.children[0].textContent.includes('No actions yet')) ul.innerHTML = '';
  const rStr = reward != null
    ? `<span class="${reward >= 0 ? 'log-reward-pos' : 'log-reward-neg'}">${reward >= 0 ? '+' : ''}${reward.toFixed(4)}</span>`
    : '';
  const eStr = error ? `<span class="log-error"> [${error}]</span>` : '';
  const sStr = (slaViolations && slaViolations.length)
    ? `<br><span style="color:var(--orange);font-size:11px">SLA: ${slaViolations.join(', ')}</span>` : '';
  const li = document.createElement('li');
  li.innerHTML = `<span class="log-time">${timeStr()}</span><span class="log-action">${action}</span> ${rStr}${eStr}${sStr}`;
  ul.prepend(li);
}

// --- Simulation Panel ------------------------------------------------------
function renderSim(sim) {
  const box = $('simBox');
  if (!sim) {
    box.innerHTML = '<p class="empty-msg" style="padding:12px 0">Click "Simulate Next" to preview projected outcome.</p>';
    return;
  }
  const safeLabel = sim.safe_to_apply ? 'Safe to Apply' : 'Not Safe';
  const safeCls = sim.safe_to_apply ? 'safe-yes' : 'safe-no';
  box.innerHTML = `
    <div class="sim-row"><span class="sim-label">Projected Cost / hr</span><b style="color:var(--accent)">$${(sim.projected_cost_per_hour||0).toFixed(4)}</b></div>
    <div class="sim-row"><span class="sim-label">Budget Remaining</span><b style="color:var(--green)">$${(sim.projected_budget_remaining||0).toFixed(4)}</b></div>
    <div class="sim-row"><span class="sim-label">Projected Reward</span><b class="${(sim.projected_reward||0)>=0?'log-reward-pos':'log-reward-neg'}">${(sim.projected_reward||0).toFixed(4)}</b></div>
    <div class="sim-row"><span class="sim-label">SLA Violations</span><b style="color:var(--red)">${(sim.projected_sla_violations||[]).join(', ')||'None'}</b></div>
    <div class="sim-row"><span class="sim-label">Cascade Risks</span><b style="color:var(--yellow)">${(sim.cascading_risks||[]).join(', ')||'None'}</b></div>
    <div class="sim-row"><span class="sim-label">Decision</span><b class="${safeCls}">${safeLabel}</b></div>
    ${sim.recommendation ? `<div class="sim-note">${sim.recommendation}</div>` : ''}
  `;
}

// --- Full Refresh ----------------------------------------------------------
async function refreshState() {
  const res = await fetch('/state').catch(() => null);
  if (!res || !res.ok) return {};
  return await res.json();
}

function applyObs(obs, state) {
  stepCount = obs.step_count || 0;
  cascadeRisks = obs.cascading_risks || {};
  setStatus(
    `Task: ${obs.task_id || '--'}  |  Step ${stepCount} / ${obs.max_steps || 20}  |  Budget remaining: $${(obs.budget_remaining||0).toFixed(4)}`,
    'var(--accent)'
  );
  renderResources(obs.resources || [], obs.sla_violations || []);
  renderCost(obs, state || {});
  renderDepGraph(obs.resources || [], obs.dependency_graph || {}, obs.cascading_risks || {}, obs.sla_violations || []);
}

// --- Actions ---------------------------------------------------------------
// --- Helper: check episode is active before step/simulate/grade
function episodeActive() {
  const status = $('episodeStatus').textContent || '';
  return status.includes('Task:');
}

async function safeJson(r) {
  // Returns parsed JSON or throws a friendly error
  const text = await r.text();
  try { return JSON.parse(text); }
  catch(e) { throw new Error('Server error ' + r.status + ': ' + text.slice(0, 120)); }
}

async function doReset() {
  loading(true);
  initialCost = null; stepCount = 0; cascadeRisks = {};
  if (costChart) { costChart.data.labels = []; costChart.data.datasets[0].data = []; costChart.update(); }
  const task = $('taskSelect').value;
  try {
    const r = await fetch('/reset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ task_id: task })
    });
    const obs = await safeJson(r);
    if (obs.detail) throw new Error(obs.detail);
    const state = await refreshState();
    applyObs(obs, state);
    addLog('reset -- ' + task, null, null, null);
    renderSim(null);
  } catch(e) { addLog('reset failed', null, String(e), null); }
  finally { loading(false); }
}

async function doStep() {
  if (!episodeActive()) {
    addLog('step skipped', null, 'No active episode -- click Reset first', null);
    return;
  }
  loading(true);
  let action = { action_type: 'noop' };
  const rows = document.querySelectorAll('#resBody tr:not(.terminated)');
  let picked = null;
  rows.forEach(row => {
    const cells = row.querySelectorAll('td');
    if (!picked && cells[5]) {
      const status = cells[5].textContent.trim();
      const critical = cells[0].querySelector('.pill-red');
      if (status === 'idle' && !critical) {
        const sp = cells[0].querySelector('span');
        if (sp) picked = sp.textContent.trim();
      }
    }
  });
  if (picked) action = { action_type: 'terminate', resource_id: picked };
  try {
    const r = await fetch('/step', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action })
    });
    const obs = await safeJson(r);
    if (obs.detail) throw new Error(obs.detail);
    const state = await refreshState();
    applyObs(obs, state);
    const reward = obs.reward != null ? obs.reward : null;
    addLog(`${action.action_type}${action.resource_id ? ' -- ' + action.resource_id : ''}`, reward, obs.last_action_error, obs.sla_violations);
    if (obs.done) setStatus('Episode complete -- click Reset to start again', 'var(--green)');
  } catch(e) { addLog('step failed', null, String(e), null); }
  finally { loading(false); }
}

async function doSimulate() {
  if (!episodeActive()) {
    addLog('simulate skipped', null, 'No active episode -- click Reset first', null);
    return;
  }
  loading(true);
  let resource_id = null;
  document.querySelectorAll('#resBody tr:not(.terminated)').forEach(row => {
    if (!resource_id) {
      const cells = row.querySelectorAll('td');
      if (cells[5] && cells[5].textContent.trim() === 'idle') {
        const sp = cells[0].querySelector('span');
        if (sp) resource_id = sp.textContent.trim();
      }
    }
  });
  const body = resource_id ? { action_type: 'terminate', resource_id } : { action_type: 'noop' };
  try {
    const r = await fetch('/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    if (r.ok) renderSim(await r.json());
    else { const err = await r.json().catch(() => ({})); renderSim(null); addLog('simulate', null, err.detail || 'Server error ' + r.status, null); }
  } catch(e) { renderSim(null); addLog('simulate failed', null, String(e), null); }
  finally { loading(false); }
}

async function doGrade() {
  if (!episodeActive()) {
    addLog('grade skipped', null, 'No active episode -- click Reset first', null);
    return;
  }
  loading(true);
  try {
    const r = await fetch('/grade');
    if (r.ok) {
      const g = await r.json();
      const score = g.score != null ? g.score : g;
      addLog('Grade', parseFloat(score), null, null);
      setStatus(`Score: ${parseFloat(score).toFixed(4)} -- Task: ${$('taskSelect').value}`, 'var(--purple)');
    } else {
      const err = await r.json().catch(() => ({}));
      addLog('grade failed', null, err.detail || 'Server error ' + r.status, null);
    }
  } finally { loading(false); }
}

// --- Init ------------------------------------------------------------------
initChart();
</script>
</body></html>
"""


@app.get("/web", response_class=HTMLResponse, include_in_schema=False)
async def web_ui():
    """Serve the interactive Cloud FinOps Optimizer Web UI."""
    return HTMLResponse(content=_WEB_UI)

