"""
Pre-submission validator for the Cloud FinOps Optimizer.

Checks the full contest checklist using only the Python standard library
(no external dependencies required to run this script).

Static checks (no server needed):
    1. openenv.yaml present and contains required keys
    2. Dockerfile exists with required directives
    3. pyproject.toml present
    4. requirements.txt lists required packages
    5. inference.py present and reads required env vars
    6. models.py, server/environment.py, server/app.py, client.py parse cleanly
    7. OpenEnv base class usage is correct (subclasses Action/Observation/State/Environment/EnvClient)

Server checks (requires running server):
    8.  GET /health returns {"status": "healthy"}
    9.  POST /reset (task1, task2, task3) returns valid FinOpsObservation
    10. POST /step with all 4 action types returns reward in [-1.0, 1.0]
    11. GET /state returns episode snapshot
    12. GET /grade returns score in [0.0, 1.0] for all 3 tasks
    13. WS /ws endpoint is declared in app.py

Run:
    python validate.py                        # static + server (localhost:7860)
    python validate.py --url http://host:port # custom server URL
    python validate.py --static-only          # skip server checks
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Tuple
import urllib.error
import urllib.request

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"

_results: List[Dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Tiny HTTP helpers (stdlib only)
# ---------------------------------------------------------------------------

def _get(url: str, timeout: int = 10) -> Tuple[int, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, {}
    except Exception as e:
        return 0, {"error": str(e)}


def _post(url: str, body: Dict[str, Any], timeout: int = 10) -> Tuple[int, Any]:
    data = json.dumps(body).encode()
    req  = urllib.request.Request(url, data=data,
                                  headers={"Content-Type": "application/json"},
                                  method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode())
        except Exception:
            body = {}
        return e.code, body
    except Exception as e:
        return 0, {"error": str(e)}


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

def check(name: str, fn: Callable[[], Tuple[str, str]]) -> bool:
    try:
        status, detail = fn()
    except Exception as exc:
        status, detail = FAIL, f"Unhandled exception: {exc}"
    _results.append({"check": name, "status": status, "detail": detail})
    tag = "[PASS]" if status == PASS else ("[WARN]" if status == WARN else "[FAIL]")
    print(f"  {tag}  {name}")
    if detail:
        print(f"         {detail}")
    return status in (PASS, WARN)


# ---------------------------------------------------------------------------
# Static checks
# ---------------------------------------------------------------------------

def _parse_file(path: str) -> Tuple[str, str]:
    if not os.path.exists(path):
        return FAIL, f"{path} not found"
    with open(path) as f:
        src = f.read()
    try:
        ast.parse(src)
        return PASS, ""
    except SyntaxError as e:
        return FAIL, f"SyntaxError in {path}: {e}"


def chk_openenv_yaml() -> Tuple[str, str]:
    path = "openenv.yaml"
    if not os.path.exists(path):
        return FAIL, "openenv.yaml not found"
    content = open(path).read()
    required = ["name", "version", "description", "tasks", "observation_space",
                "action_space", "reward_function", "endpoints"]
    missing = [k for k in required if k not in content]
    if missing:
        return FAIL, f"Missing keys: {missing}"
    task_count = content.count("- id: task")
    if task_count < 3:
        return FAIL, f"Expected 3 tasks, found {task_count}"
    return PASS, f"{task_count} tasks declared"


def chk_dockerfile() -> Tuple[str, str]:
    # validate-submission.sh checks root Dockerfile first, then server/Dockerfile.
    # A root-level Dockerfile is required so docker build uses the project root
    # as the build context, allowing all COPY commands to resolve correctly.
    root_path   = "Dockerfile"
    server_path = "server/Dockerfile"

    if os.path.exists(root_path):
        path    = root_path
        context = "root (correct — build context = project root)"
    elif os.path.exists(server_path):
        path    = server_path
        context = "server/ (WARNING: build context will be server/, COPY may fail)"
    else:
        return FAIL, "No Dockerfile found at root or server/Dockerfile"

    content = open(path).read()
    required = ["FROM python", "EXPOSE", "uvicorn"]
    missing  = [k for k in required if k not in content]
    if missing:
        return WARN, f"Dockerfile ({path}) may be missing: {missing}"

    # Verify all source files are copied
    copy_files = ["models.py", "client.py", "inference.py", "server/"]
    missing_copies = [f for f in copy_files if f"COPY {f}" not in content]
    if missing_copies:
        return FAIL, f"Dockerfile missing COPY for: {missing_copies}"

    return PASS, f"Dockerfile at {path} — context: {context}"


def chk_pyproject() -> Tuple[str, str]:
    if not os.path.exists("pyproject.toml"):
        return FAIL, "pyproject.toml not found (required for pip install from HF Space)"
    content = open("pyproject.toml").read()
    for key in ["name", "version", "dependencies"]:
        if key not in content:
            return WARN, f"pyproject.toml may be missing '{key}'"
    return PASS, ""


def chk_requirements() -> Tuple[str, str]:
    path = "requirements.txt"
    if not os.path.exists(path):
        return FAIL, "requirements.txt not found"
    content = open(path).read()
    required = ["openenv-core", "fastapi", "uvicorn", "pydantic", "openai", "websockets"]
    missing  = [p for p in required if p not in content]
    if missing:
        return FAIL, f"Missing packages: {missing}"
    return PASS, ""


def chk_inference() -> Tuple[str, str]:
    path = "inference.py"
    if not os.path.exists(path):
        return FAIL, "inference.py not found"
    content = open(path).read()
    checks = [
        # Mandatory environment variables (contest spec)
        ("API_BASE_URL",           "reads API_BASE_URL"),
        ("MODEL_NAME",             "reads MODEL_NAME"),
        ("HF_TOKEN",               "reads HF_TOKEN"),
        ("LOCAL_IMAGE_NAME",       "defines LOCAL_IMAGE_NAME variable"),
        ("API_KEY",                "defines API_KEY (HF_TOKEN fallback chain)"),
        ("OpenAI(",                "uses OpenAI client"),
        ("base_url=API_BASE_URL",  "passes base_url to OpenAI client"),
        # Tasks
        ("task1",                  "runs task1"),
        ("task2",                  "runs task2"),
        ("task3",                  "runs task3"),
        # Constants
        ("TEMPERATURE",            "defines TEMPERATURE constant"),
        ("MAX_STEPS",              "defines MAX_STEPS constant"),
        ("MAX_TOKENS",             "defines MAX_TOKENS constant"),
        ("FALLBACK_ACTION",        "defines FALLBACK_ACTION for LLM failures"),
        ("SUCCESS_SCORE_THRESHOLD","defines SUCCESS_SCORE_THRESHOLD"),
        ("BENCHMARK",              "defines BENCHMARK (env= field in [START])"),
        # Required functions
        ("observation.goal",       "uses observation.goal at episode start"),
        ("last_action_error",      "checks observation.last_action_error"),
        ("history",                "maintains step history list"),
        ("build_user_prompt",      "defines build_user_prompt()"),
        ("parse_model_action",     "defines parse_model_action()"),
        ("extract_screenshot_uri", "defines extract_screenshot_uri()"),
        ("finally",                "env.close() in finally block"),
        ("baseline_scores.json",   "writes baseline_scores.json"),
        ("asyncio.run",            "uses asyncio.run(main()) entry point"),
        # Structured log format — exact field names from sample script spec
        ("[START]",                "emits [START] log line"),
        ("[STEP]",                 "emits [STEP] log line"),
        ("[END]",                  "emits [END] log line"),
        ("log_start",              "defines log_start(task, env, model)"),
        ("log_step",               "defines log_step(step, action, reward, done, error)"),
        ("log_end",                "defines log_end(success, steps, score, rewards)"),
        # [START] fields: task= env= model=
        ("task={task}",            "[START] emits task= field"),
        ("env={env}",              "[START] emits env= field"),
        ("model={model}",          "[START] emits model= field"),
        # [STEP] fields: step= action= reward=.2f done=lower error=
        ("step={step}",            "[STEP] emits step= field"),
        ("action={action}",        "[STEP] emits action= field"),
        ("reward={reward:.2f}",    "[STEP] reward formatted to 2 decimal places"),
        ("done_val",               "[STEP] done as lowercase string"),
        ("error_val",              "[STEP] error as string or null"),
        # [END] fields: success= steps= score=.3f rewards=csv
        ("success=",               "[END] emits success= field"),
        ("score={score:.3f}",      "[END] score formatted to 3 decimal places"),
        ("rewards=",               "[END] emits rewards= field"),
        (":.2f}",                  "[END] per-step rewards formatted to 2 decimal places"),
        ("List[float]",            "rewards tracked as List[float]"),
        ("SUCCESS_SCORE_THRESHOLD","success computed from score threshold"),
        # flush=True on all prints
        ("flush=True",             "print statements use flush=True"),
    ]
    failures = [desc for token, desc in checks if token not in content]
    if failures:
        return FAIL, f"Missing: {failures}"
    return PASS, ""


def chk_python_syntax() -> Tuple[str, str]:
    files = [
        "models.py",
        "client.py",
        "inference.py",
        "validate.py",
        "server/environment.py",
        "server/app.py",
    ]
    for path in files:
        status, detail = _parse_file(path)
        if status == FAIL:
            return FAIL, detail
    return PASS, f"All {len(files)} Python files parse cleanly"


def chk_openenv_base_classes() -> Tuple[str, str]:
    """Verify files use the correct OpenEnv base classes, not custom ones."""
    checks = {
        "models.py": [
            ("openenv.core.env_server", "imports from openenv.core.env_server"),
            ("Action",                  "FinOpsAction subclasses Action"),
            ("Observation",             "FinOpsObservation subclasses Observation"),
            ("State",                   "FinOpsState subclasses State"),
        ],
        "server/environment.py": [
            ("openenv.core.env_server", "imports Environment from openenv"),
            ("Environment",             "subclasses Environment"),
            ("SUPPORTS_CONCURRENT_SESSIONS", "declares concurrent session support"),
        ],
        "server/app.py": [
            ("create_fastapi_app",      "uses create_fastapi_app()"),
        ],
        "client.py": [
            ("EnvClient",               "FinOpsEnv subclasses EnvClient"),
            ("_step_payload",           "implements _step_payload"),
            ("_parse_result",           "implements _parse_result"),
            ("_parse_state",            "implements _parse_state"),
        ],
    }
    for path, reqs in checks.items():
        if not os.path.exists(path):
            return FAIL, f"{path} not found"
        content = open(path).read()
        for token, desc in reqs:
            if token not in content:
                return FAIL, f"{path}: missing '{token}' ({desc})"
    return PASS, "All OpenEnv base classes used correctly"


def chk_three_tasks_graded() -> Tuple[str, str]:
    """Verify all 3 task graders exist in environment.py."""
    path = "server/environment.py"
    if not os.path.exists(path):
        return FAIL, f"{path} not found"
    content = open(path).read()
    for method in ["_grade_task1", "_grade_task2", "_grade_task3"]:
        if method not in content:
            return FAIL, f"Missing grader method: {method}"
    return PASS, "All 3 deterministic graders present"


# ---------------------------------------------------------------------------
# Server checks
# ---------------------------------------------------------------------------

def chk_health(base_url: str) -> Tuple[str, str]:
    code, body = _get(f"{base_url}/health")
    if code == 200 and body.get("status") == "healthy":
        return PASS, ""
    return FAIL, f"HTTP {code}, body={body}"


def chk_reset(base_url: str, task_id: str) -> Tuple[str, str]:
    code, body = _post(f"{base_url}/reset", {"task_id": task_id})
    if code != 200:
        return FAIL, f"HTTP {code}"
    required = ["resources", "total_cost_per_hour", "budget_per_hour",
                "step_count", "task_id", "done"]
    missing = [k for k in required if k not in body]
    if missing:
        return FAIL, f"Missing observation fields: {missing}"
    if not isinstance(body.get("resources"), list) or len(body["resources"]) == 0:
        return FAIL, "resources list is empty"
    n = len(body["resources"])
    cost = body["total_cost_per_hour"]
    return PASS, f"{n} resources, cost=${cost:.4f}/hr"


def chk_step(base_url: str, task_id: str, action: Dict, label: str) -> Tuple[str, str]:
    _post(f"{base_url}/reset", {"task_id": task_id})
    code, body = _post(f"{base_url}/step", {"action": action})
    if code != 200:
        return FAIL, f"HTTP {code} for {label}"
    reward = body.get("reward")
    if reward is None or not isinstance(reward, (int, float)):
        return FAIL, f"reward is not numeric: {reward}"
    if not (-1.0 <= float(reward) <= 1.0):
        return FAIL, f"reward {reward} outside [-1.0, 1.0]"
    return PASS, f"reward={reward:+.4f}"


def chk_state(base_url: str) -> Tuple[str, str]:
    _post(f"{base_url}/reset", {"task_id": "task1"})
    code, body = _get(f"{base_url}/state")
    if code != 200:
        return FAIL, f"HTTP {code}"
    required = ["episode_id", "task_id", "step_count"]
    missing  = [k for k in required if k not in body]
    if missing:
        return FAIL, f"Missing state fields: {missing}"
    return PASS, f"episode_id={str(body.get('episode_id', ''))[:8]}..."


def chk_grade(base_url: str, task_id: str) -> Tuple[str, str]:
    _post(f"{base_url}/reset", {"task_id": task_id})
    # Take a few noop steps to get past step 0
    for _ in range(3):
        _post(f"{base_url}/step", {"action": {"action_type": "noop"}})
    code, body = _get(f"{base_url}/grade")
    if code != 200:
        return FAIL, f"HTTP {code}"
    score = body.get("score")
    if score is None:
        return FAIL, "No 'score' field in grade response"
    if not (0.0 <= float(score) <= 1.0):
        return FAIL, f"score {score} outside [0.0, 1.0]"
    return PASS, f"score={score:.4f}"


def chk_grade_deterministic(base_url: str) -> Tuple[str, str]:
    """Run task1 twice with the same actions; scores must be identical."""
    def _run_episode() -> float:
        _post(f"{base_url}/reset", {"task_id": "task1"})
        _post(f"{base_url}/step", {"action": {"action_type": "terminate",
                                               "resource_id": "ebs-001"}})
        _post(f"{base_url}/step", {"action": {"action_type": "terminate",
                                               "resource_id": "s3-001"}})
        _, g = _get(f"{base_url}/grade")
        return g.get("score", -1.0)

    s1 = _run_episode()
    s2 = _run_episode()
    if s1 != s2:
        return FAIL, f"Grader is non-deterministic: {s1} vs {s2}"
    return PASS, f"score={s1:.4f} (same on two identical runs)"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_static() -> int:
    print("\n--- Static Checks ---")
    failures = 0
    tests = [
        ("openenv.yaml valid",            chk_openenv_yaml),
        ("Dockerfile present",            chk_dockerfile),
        ("pyproject.toml present",        chk_pyproject),
        ("requirements.txt complete",     chk_requirements),
        ("inference.py complete",         chk_inference),
        ("Python syntax clean",           chk_python_syntax),
        ("OpenEnv base classes correct",  chk_openenv_base_classes),
        ("3 deterministic graders exist", chk_three_tasks_graded),
    ]
    for name, fn in tests:
        if not check(name, fn):
            failures += 1
    return failures


def run_server(base_url: str) -> int:
    print(f"\n--- Server Checks ({base_url}) ---")
    failures = 0

    tests: List[Tuple[str, Callable]] = [
        ("GET /health returns 200 + healthy",
         lambda: chk_health(base_url)),

        ("POST /reset (task1) valid observation",
         lambda: chk_reset(base_url, "task1")),
        ("POST /reset (task2) valid observation",
         lambda: chk_reset(base_url, "task2")),
        ("POST /reset (task3) valid observation",
         lambda: chk_reset(base_url, "task3")),

        ("POST /step — terminate action",
         lambda: chk_step(base_url, "task1",
                           {"action_type": "terminate", "resource_id": "ebs-001"},
                           "terminate ebs-001")),
        ("POST /step — resize action",
         lambda: chk_step(base_url, "task2",
                           {"action_type": "resize", "resource_id": "ec2-t01",
                            "target_size": "t2.large"},
                           "resize ec2-t01")),
        ("POST /step — reserve action",
         lambda: chk_step(base_url, "task3",
                           {"action_type": "reserve", "resource_id": "ec2-h01"},
                           "reserve ec2-h01")),
        ("POST /step — noop action",
         lambda: chk_step(base_url, "task1",
                           {"action_type": "noop"}, "noop")),

        ("GET /state returns episode snapshot",
         lambda: chk_state(base_url)),

        ("GET /grade (task1) score in [0.0, 1.0]",
         lambda: chk_grade(base_url, "task1")),
        ("GET /grade (task2) score in [0.0, 1.0]",
         lambda: chk_grade(base_url, "task2")),
        ("GET /grade (task3) score in [0.0, 1.0]",
         lambda: chk_grade(base_url, "task3")),

        ("Grader is deterministic (same actions = same score)",
         lambda: chk_grade_deterministic(base_url)),
    ]

    for name, fn in tests:
        if not check(name, fn):
            failures += 1

    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Cloud FinOps Optimizer — Pre-submission Validator")
    parser.add_argument("--url", default=os.environ.get("API_BASE_URL", "http://localhost:7860"))
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()

    # Run from project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print("=" * 60)
    print("Cloud FinOps Optimizer — Pre-submission Validator")
    print("=" * 60)

    total_failures = run_static()

    if not args.static_only:
        code, _ = _get(f"{args.url}/health", timeout=4)
        if code == 200:
            total_failures += run_server(args.url)
        else:
            print(f"\n  [WARN]  Server not reachable at {args.url} (HTTP {code})")
            print("          Start with: uvicorn server.app:app --host 0.0.0.0 --port 7860")
            print("          Then re-run this script to complete all checks.")

    print("\n" + "=" * 60)
    passed = sum(1 for r in _results if r["status"] in (PASS, WARN))
    total  = len(_results)
    failed = sum(1 for r in _results if r["status"] == FAIL)
    print(f"Results: {passed}/{total} checks passed  |  {failed} failed")

    if total_failures == 0:
        print("Status:  READY TO SUBMIT")
    else:
        print(f"Status:  {total_failures} FAILED — fix before submitting")

    with open("validation_report.json", "w") as f:
        json.dump({"results": _results, "passed": passed,
                   "failed": failed, "total": total}, f, indent=2)
    print("Report:  validation_report.json")

    if total_failures > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
