"""
Baseline inference script for the Cloud FinOps Optimizer.

STDOUT FORMAT (mandatory — parsed by automated evaluator):

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Rules (from sample inference script spec):
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards formatted to 2 decimal places.
    - score formatted to 3 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each task must return score in (0, 1).

Mandatory environment variables:
    API_BASE_URL      The API endpoint for the LLM / environment server.
    MODEL_NAME        The model identifier to use for inference.
    HF_TOKEN          Your Hugging Face / API key.
    LOCAL_IMAGE_NAME  Name of the local Docker image (if using from_docker_image()).
"""

import asyncio
import json
import os
import sys
import time
import urllib.request
from typing import Any, Dict, List, Optional

from openai import OpenAI

from client import FinOpsEnv, FinOpsAction

# Configuration — all read from environment variables per spec
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("ERROR: HF_TOKEN environment variable is required", file=sys.stderr)
    sys.exit(1)

# API key: per guidelines, we use HF_TOKEN as the API key
API_KEY = HF_TOKEN

# Benchmark / environment name used in [START] env= field
BENCHMARK = "cloud-finops-optimizer"

MAX_STEPS  = 20
MAX_TOKENS = 400
TEMPERATURE = 0.0

# score >= this threshold counts as success in [END] success= field
SUCCESS_SCORE_THRESHOLD = 0.1

FALLBACK_ACTION = '{"action_type": "noop", "reasoning": "LLM unavailable, defaulting to noop."}'

TASKS = ["task1", "task2", "task3"]

# ---------------------------------------------------------------------------
# Structured log functions — exact signatures and format from sample script
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    """
    [START] task=<task_name> env=<benchmark> model=<model_name>
    One line at episode begin.
    """
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    """
    [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    One line immediately after each env.step() returns.
    - reward: 2 decimal places
    - done: lowercase boolean string
    - error: raw error string or "null"
    """
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    """
    [END] success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
    One line after env.close(), always emitted even on exception.
    - success: lowercase boolean string
    - rewards: comma-separated per-step rewards, each at 2 decimal places
    """
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert Cloud FinOps engineer acting as an AI agent.
Your goal is to reduce cloud infrastructure costs safely.

WORKFLOW (follow this every step):
1. Read the observation: resources, dependency_graph, cascading_risks, sla_violations.
2. Propose an action using simulate FIRST to check safety before committing.
3. Read the simulate_result recommendation.
4. If safe_to_apply is true, commit the real action with your reasoning.
   If false, pick a safer action or use simulate on an alternative.

Available actions — respond with exactly one JSON object, no markdown:

  Simulate (preview without changing state):
    {"action_type": "simulate",
     "simulate_action": {"action_type": "terminate", "resource_id": "<id>"},
     "reasoning": "<why you want to check this>"}

  Terminate an idle resource:
    {"action_type": "terminate", "resource_id": "<id>",
     "reasoning": "<cost saving and safety justification>"}

  Resize an over-provisioned EC2 instance (target must be smaller):
    {"action_type": "resize", "resource_id": "<id>", "target_size": "<size>",
     "reasoning": "<SLA check and cost justification>"}
    Sizes (ascending): t2.micro t2.small t2.medium t2.large t2.xlarge t2.2xlarge

  Apply reserved pricing (40% discount, permanent):
    {"action_type": "reserve", "resource_id": "<id>",
     "reasoning": "<utilisation and cost justification>"}

  No action:
    {"action_type": "noop", "reasoning": "<why no action is best>"}

RULES:
- NEVER terminate critical=true resources
- NEVER resize if projected CPU will exceed sla_max_cpu
- Check cascading_risks before terminating any resource
- Use simulate before every non-trivial action
- Always include a reasoning field

JSON only. No explanation outside the JSON object."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_user_prompt(step: int, observation: Any, history: List[str]) -> str:
    """Build the per-step user prompt including full resource, dependency and SLA context."""
    active = [r for r in observation.resources if r.status != "terminated"]
    lines = []
    for r in active:
        size     = f" ({r.instance_size})" if r.instance_size else ""
        cooldown = f" [cooldown:{r.resize_cooldown_steps}]" if r.resize_cooldown_steps else ""
        sla      = f" SLA:{r.sla_status}" if r.sla_status != "ok" else ""
        lines.append(
            f"  {r.id:12s}  type={r.type}{size:16s}"
            f"  cpu={r.cpu_utilization:5.1f}% (max={r.sla_max_cpu:.0f}%)"
            f"  ${r.cost_per_hour:.4f}/hr"
            f"  status={r.status}"
            f"  critical={r.critical}"
            f"  reserved={r.reserved}"
            f"  idle_hrs={r.idle_hours}"
            f"{cooldown}{sla}"
        )

    dep_lines  = [f"  {rid} depends on: {deps}"
                  for rid, deps in (observation.dependency_graph or {}).items() if deps]
    risk_lines = [f"  removing {rid} cascades to: {risks}"
                  for rid, risks in (observation.cascading_risks or {}).items() if risks]

    sla_block = ""
    if getattr(observation, "sla_violations", []):
        sla_block = f"\nACTIVE SLA VIOLATIONS: {observation.sla_violations}"

    sim_block = ""
    if getattr(observation, "simulate_result", None):
        sr = observation.simulate_result
        sim_block = (
            f"\nLAST SIMULATE RESULT:"
            f"\n  Action:      {sr.proposed_action}"
            f"\n  Safe:        {sr.safe_to_apply}"
            f"\n  Proj cost:   ${sr.projected_cost_per_hour:.4f}/hr"
            f"\n  Proj reward: {sr.projected_reward:+.4f}"
            f"\n  Advice:      {sr.recommendation}"
        )

    history_block = "\n".join(history[-6:]) if history else "  (no history yet)"
    dep_section   = ("\nDependency graph:\n" + "\n".join(dep_lines))  if dep_lines  else ""
    risk_section  = ("\nCascade risks:\n"    + "\n".join(risk_lines)) if risk_lines else ""

    return (
        f"Step {step} of {observation.max_steps}\n"
        f"Total cost/hr   : ${observation.total_cost_per_hour:.4f}\n"
        f"Budget remaining: ${observation.budget_remaining:.4f}\n"
        f"{sla_block}"
        f"\nActive resources:\n" + "\n".join(lines) +
        dep_section + risk_section + sim_block +
        f"\n\nAction history:\n{history_block}"
    )


def extract_screenshot_uri(observation: Any) -> Optional[str]:
    """Text-only environment — always returns None."""
    return None


def parse_model_action(response_text: str) -> str:
    """Parse a JSON action string from model output. Returns FALLBACK_ACTION on failure."""
    text = response_text.strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:]).rstrip("`").strip()
    try:
        obj = json.loads(text)
        if "action_type" not in obj:
            return FALLBACK_ACTION
        return json.dumps(obj)
    except json.JSONDecodeError:
        print(f"[DEBUG] Could not parse model output: {response_text!r}", flush=True)
        return FALLBACK_ACTION


def _fetch_grade(base_url: str) -> Dict[str, Any]:
    """Fetch episode grade from /grade endpoint. Returns {score: 0.0} on failure."""
    try:
        with urllib.request.urlopen(f"{base_url}/grade", timeout=10) as r:
            return json.loads(r.read().decode())
    except Exception as exc:
        print(f"[DEBUG] grade fetch failed: {exc}", flush=True)
        return {"score": 0.01, "error": str(exc)}


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(env: Any, client: OpenAI, task_id: str) -> Dict[str, Any]:
    """
    Run one full episode. Emits [START], [STEP]×N, [END] log lines.
    Returns result dict for the summary table.
    """
    history:  List[str]   = []
    rewards:  List[float] = []   # per-step rewards — passed to log_end
    steps_taken = 0
    success     = False

    result      = env.reset(task_id=task_id)
    observation = result.observation

    # ── [START] ───────────────────────────────────────────────────────────
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    print(f"Episode goal: {observation.goal}", flush=True)

    try:
        for step in range(1, MAX_STEPS + 1):
            if result.done:
                print("Environment signalled done. Stopping early.", flush=True)
                break

            user_prompt  = build_user_prompt(step, observation, history)
            user_content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]

            screenshot_uri = extract_screenshot_uri(observation)
            if screenshot_uri:
                user_content.append(
                    {"type": "image_url", "image_url": {"url": screenshot_uri}}
                )

            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user",   "content": user_content},
            ]

            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=False,
                )
                response_text = completion.choices[0].message.content or ""
            except Exception as exc:
                print(f"[DEBUG] Model request failed ({exc}). Using fallback.", flush=True)
                response_text = FALLBACK_ACTION

            action_str  = parse_model_action(response_text)
            action_dict = json.loads(action_str)

            print(f"Step {step}: model suggested -> {action_str}", flush=True)

            action = FinOpsAction(
                action_type=action_dict.get("action_type", "noop"),
                resource_id=action_dict.get("resource_id"),
                target_size=action_dict.get("target_size"),
                simulate_action=action_dict.get("simulate_action"),
                reasoning=action_dict.get("reasoning"),
            )

            result      = env.step(action)
            observation = result.observation
            reward      = result.reward or 0.0
            done        = result.done
            error       = observation.last_action_error

            rewards.append(reward)
            steps_taken = step

            # ── [STEP] ────────────────────────────────────────────────────
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            history.append(
                f"Step {step}: {action_str} -> reward {reward:+.2f}"
                + (" ERROR" if error else "")
            )
            print(
                f"  Reward: {reward:+.2f} | Done: {done} | Error: {error}",
                flush=True,
            )

            if done:
                print("Episode complete.", flush=True)
                break

        else:
            print(f"Reached max steps ({MAX_STEPS}).", flush=True)

    finally:
        grade   = _fetch_grade(API_BASE_URL)
        # Phase 2 Compliance: ensure inference script log is strictly (0, 1)
        score   = min(max(float(grade.get("score", 0.01)), 0.01), 0.99)
        success = score >= SUCCESS_SCORE_THRESHOLD
        env.close()

        # ── [END] — always emitted, even on exception ─────────────────────
        log_end(success=success, steps=steps_taken, rewards=rewards)

    print(
        f"  Score: {score:.4f}  |  "
        f"Explainability: {grade.get('explainability_score', 0.0):.4f}",
        flush=True,
    )

    return {
        "task_id":     task_id,
        "score":       score,
        "rewards":     rewards,
        "steps_taken": steps_taken,
        "grade":       grade,
    }


# ---------------------------------------------------------------------------
# Entry point — sync wrapper matching sample pattern
# ---------------------------------------------------------------------------

async def main() -> None:
    print("Cloud FinOps Optimizer — Baseline Inference", flush=True)
    print(f"  Server : {API_BASE_URL}", flush=True)
    print(f"  Model  : {MODEL_NAME}", flush=True)

    # Health check — verify the environment server is reachable
    try:
        with urllib.request.urlopen(
            f"{API_BASE_URL}/health".replace(
                "https://router.huggingface.co/v1", "http://localhost:7860"
            ),
            timeout=5,
        ) as r:
            health = json.loads(r.read().decode())
        print(f"  Health : {health.get('status', 'unknown')}", flush=True)
    except Exception as exc:
        # For HF inference router base URLs, health check hits env server directly
        env_url = os.getenv("ENV_BASE_URL", "http://localhost:7860")
        try:
            with urllib.request.urlopen(f"{env_url}/health", timeout=5) as r:
                health = json.loads(r.read().decode())
            print(f"  Health : {health.get('status', 'unknown')}", flush=True)
        except Exception:
            print(f"  Health : unreachable ({exc})", flush=True)

    client  = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    results: List[Dict[str, Any]] = []
    start_time = time.time()

    # Each task gets its own env connection
    for task_id in TASKS:
        env = FinOpsEnv(base_url=os.getenv("ENV_BASE_URL", "http://localhost:7860")).sync()
        env.connect()
        result = run_episode(env, client, task_id)
        results.append(result)

    total_elapsed = round(time.time() - start_time, 1)

    # Summary table
    print(f"\n{'='*60}", flush=True)
    print("BASELINE SCORES", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"{'Task':<10} {'Score':>8} {'Explain':>8} {'Steps':>7} {'AvgRew':>10}", flush=True)
    print("-" * 52, flush=True)
    for r in results:
        expl    = r["grade"].get("explainability_score", 0.0)
        avg_rew = sum(r["rewards"]) / len(r["rewards"]) if r["rewards"] else 0.0
        print(
            f"{r['task_id']:<10} {r['score']:>8.4f} {expl:>8.4f}"
            f" {r['steps_taken']:>7d} {avg_rew:>10.4f}",
            flush=True,
        )
    print("-" * 52, flush=True)
    avg = sum(r["score"] for r in results) / len(results)
    print(f"{'Average':<10} {avg:>8.4f}", flush=True)
    print(f"\nTotal elapsed: {total_elapsed}s", flush=True)

    output = {
        "model":                 MODEL_NAME,
        "server":                API_BASE_URL,
        "results":               [
            {
                "task_id":     r["task_id"],
                "score":       r["score"],
                "steps_taken": r["steps_taken"],
                "rewards":     r["rewards"],
                "grade":       r["grade"],
            }
            for r in results
        ],
        "average_score":         round(avg, 4),
        "total_elapsed_seconds": total_elapsed,
    }
    with open("baseline_scores.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Results written to baseline_scores.json", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
