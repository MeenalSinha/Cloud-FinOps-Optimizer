---
title: Cloud FinOps Optimizer
emoji: 💰
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
tags:
- openenv
- RL
short_description: OpenEnv cloud optimization environment.
---

# Cloud FinOps Optimizer

An OpenEnv-compliant reinforcement learning environment where an AI agent
acts as a Cloud FinOps Engineer. The agent must reduce cloud infrastructure
costs safely by terminating waste, right-sizing over-provisioned compute,
and applying reserved pricing — without disrupting mission-critical systems.

---

## Problem Motivation

Cloud cost management is a critical discipline. Organizations routinely
overspend 20-40% on cloud infrastructure through idle resources,
over-provisioned instances, and missed pricing opportunities such as
reserved instances.

Human FinOps engineers must reason under constraints, respect operational
boundaries, and make sequential decisions with downstream consequences.
This environment models that decision-making process faithfully, making it
a high-value benchmark for training and evaluating AI agents on real-world
planning tasks.

---

## Project Structure

```
cloud-finops-env/
├── models.py              Typed Pydantic models (FinOpsAction, FinOpsObservation, FinOpsState)
├── client.py              EnvClient subclass (WebSocket + HTTP)
├── inference.py           Baseline inference script (uses OpenAI client)
├── openenv.yaml           OpenEnv metadata manifest
├── pyproject.toml         Package metadata (pip-installable from HF Space)
├── requirements.txt       Python dependencies
├── validate.py            Pre-submission validator
├── README.md              This file
└── server/
    ├── app.py             FastAPI server (create_fastapi_app)
    ├── environment.py     Core environment logic and graders
    └── Dockerfile         Container definition
```

---

## Action Space

The agent selects one action per step. All actions serialize to a JSON object.

| Action    | Description                                      | Required fields                          |
|-----------|--------------------------------------------------|------------------------------------------|
| terminate | Delete an idle or unused resource                | resource_id                              |
| resize    | Downgrade an EC2 instance (target must be smaller)| resource_id, target_size                 |
| reserve   | Apply 40% reserved-instance discount             | resource_id                              |
| simulate  | **[Premium]** Preview outcome without acting     | simulate_action (Dict)                   |
| noop      | Take no action                                   | none                                     |

> **Note on Reasoning:** Every action accepts an optional `reasoning` field. High-quality justifications significantly improve the episode grade.

---

## Observation Space

Each step returns a `FinOpsObservation` with the following fields:

| Field                | Type            | Description                                   |
|----------------------|-----------------|-----------------------------------------------|
| total_cost_per_hour  | float           | Current total hourly spend                    |
| budget_remaining     | float           | Headroom before reaching budget limit         |
| **dependency_graph** | dict            | Full resource dependency map                  |
| **cascading_risks**  | list            | Resources at risk if a dependency is cut      |
| **sla_violations**   | list            | Resources currently breaching performance SLAs|
| **simulate_result**  | object          | Outcome data from the last `simulate` action  |
| **info.finops_metrics**| dict          | Pre-calculated waste & savings potential      |

Each `CloudResource` has advanced metadata:

| Field              | Type            | Description                                    |
|--------------------|-----------------|------------------------------------------------|
| sla_max_cpu        | float           | CPU threshold (breached during resizes)        |
| sla_status         | str             | ok / at_risk / violated                        |
| cooldown_steps     | int             | Steps remaining for a resize to take effect    |

---

## Reward Function

| Component           | Value           | Trigger                                           |
|---------------------|-----------------|---------------------------------------------------|
| cost_reduction      | +0.1 to +0.5    | Lowering hourly spend                             |
| sla_penalty         | -0.20           | Violating a resource's performance SLA            |
| cascade_penalty     | -0.20 per dep   | Breaking a dependency chain                       |
| reasoning_bonus     | (in grade)      | High-quality, justifiable decisions               |
| failure_penalty     | -1.00           | Terminating critical nodes or massive outages     |

---

## Tasks

### Task 1 — Waste Cleanup (Easy)
Identify and terminate the 4 idle resources hidden among active ones. Focus on `cpu_utilization == 0.0`.

### Task 2 — Performance-Aware Rightsizing (Medium)
Downsize 5 over-provisioned EC2 instances. **Challenge**: Resizing too aggressively will breach `sla_max_cpu` constraints.

### Task 3 — Fleet-Wide Tactical Strategy (Hard)
Execute a multi-stage plan: terminate waste, reserve high-load servers, and right-size others while managing **Temporal Cooldowns** (actions take time to stabilize). 

---

## Grader Quality & Explainability

Each task includes an **Episode Grader** that returns a score in the strict `(0.01, 0.99)` range.

The final score is composed of:
1.  **Task Success (90%)**: Absolute cost reduction and SLA compliance.
2.  **Explainability Bonus (10%)**: The agent's `reasoning` is evaluated for keyword depth, logical structure ("Because X, I did Y"), and contextual awareness of resource IDs.

---

## API Endpoints

| Method    | Path     | Description                                     |
|-----------|----------|-------------------------------------------------|
| WS        | /ws      | Persistent WebSocket session (used by EnvClient)|
| GET       | /health  | Health check — returns `{"status": "healthy"}` |
| POST      | /reset   | Start new episode, returns FinOpsObservation    |
| POST      | /step    | Execute action, returns FinOpsObservation       |
| GET       | /state   | Full internal state snapshot                    |
| GET       | /grade   | Deterministic episode score 0.0-1.0             |
| GET       | /tasks   | List all tasks with descriptions                |
| GET       | /web     | Interactive web UI                              |
| GET       | /docs    | Auto-generated OpenAPI documentation            |

---

## Setup and Usage

### Install from HF Space

```bash
pip install git+https://huggingface.co/spaces/username/cloud-finops-optimizer
```

### Connect to a running Space

```python
from client import FinOpsEnv, FinOpsAction

with FinOpsEnv(base_url="https://username-cloud-finops-optimizer.hf.space").sync() as env:
    result = env.reset(task_id="task1")
    result = env.step(FinOpsAction(action_type="terminate", resource_id="ebs-001"))
    print(result.reward, result.done)
```

### Run locally

```bash
pip install -r requirements.txt

# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

# Check health
curl http://localhost:7860/health

# Run baseline inference
export OPENAI_API_KEY=sk-...
python inference.py
```

### Docker

```bash
docker build -t cloud-finops-optimizer -f server/Dockerfile .
docker run -d -p 7860:7860 cloud-finops-optimizer
curl http://localhost:7860/health
```

### Deploy to HF Spaces

```bash
openenv push --repo-id username/cloud-finops-optimizer
```

### Validate before submitting

```bash
python validate.py
```

---

## Environment Variables

| Variable       | Default               | Description                              |
|----------------|-----------------------|------------------------------------------|
| API_BASE_URL   | http://localhost:7860 | Environment server URL                   |
| MODEL_NAME     | gpt-4o-mini           | LLM model for baseline inference         |
| OPENAI_API_KEY | (required)            | OpenAI-compatible API key                |
| HF_TOKEN       | (optional)            | Hugging Face token for deployment        |
| PORT           | 7860                  | Server bind port                         |
| HOST           | 0.0.0.0               | Server bind host                         |
| WORKERS        | 1                     | Uvicorn worker count                     |

---

## Baseline Scores

Scores produced by `gpt-4o-mini` at temperature=0:

| Task  | Difficulty | Score  | Notes                                       |
|-------|------------|--------|---------------------------------------------|
| task1 | Easy       | 0.7500 | Correctly identifies most idle storage      |
| task2 | Medium     | 0.5000 | Partial success on EC2 right-sizing         |
| task3 | Hard       | 0.3500 | Progress toward budget but misses optimum   |

Average: 0.5333

Scores are reproducible at temperature=0.

---

## Design Notes

**Determinism.** All resource lists are hard-coded constants, not randomly
generated. The same agent behavior always produces the same grade.

**Dense rewards.** Every action returns a non-zero signal. Cost saved per
action, idle bonuses, strategy bonuses, and penalties are applied immediately.
This prevents sparse-reward problems during RL training.

**Mission-critical safety.** Terminating a critical resource applies a -1.0
reward and is penalized in the grade. This models the real-world constraint
that FinOps actions must not disrupt production.

**Resource dependencies.** Some resources declare `dependency_ids`. Breaking
an active dependency applies an additional -0.20 penalty per affected resource,
encouraging the agent to reason about downstream effects.

**Hardware.** The environment runs on 2 vCPU / 8 GB RAM. No ML model is
loaded. Inference completes well within the 20-minute runtime limit.
