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
| resize    | Downgrade an over-provisioned EC2 instance       | resource_id, target_size                 |
| reserve   | Apply 40% reserved-instance discount             | resource_id                              |
| noop      | Take no action (small negative reward)           | none                                     |

Valid instance sizes (smallest to largest):
`t2.micro` `t2.small` `t2.medium` `t2.large` `t2.xlarge` `t2.2xlarge`

Example actions:

```json
{"action_type": "terminate", "resource_id": "ebs-001"}
{"action_type": "resize",    "resource_id": "ec2-t01", "target_size": "t2.medium"}
{"action_type": "reserve",   "resource_id": "ec2-h01"}
{"action_type": "noop"}
```

---

## Observation Space

Each step returns a `FinOpsObservation` with the following fields:

| Field                | Type            | Description                                   |
|----------------------|-----------------|-----------------------------------------------|
| done                 | bool            | Whether the episode has ended                 |
| reward               | float or null   | Reward for the last action                    |
| resources            | list            | All CloudResource objects in the environment  |
| total_cost_per_hour  | float           | Current total hourly cost (active resources)  |
| budget_per_hour      | float           | Budget target                                 |
| budget_remaining     | float           | budget_per_hour minus total_cost_per_hour     |
| task_id              | str             | Active task identifier                        |
| task_description     | str             | Human-readable task objective                 |
| step_count           | int             | Steps taken in this episode                   |
| max_steps            | int             | Step budget for this task                     |
| info                 | dict            | Auxiliary information                         |

Each `CloudResource` has:

| Field              | Type            | Description                                    |
|--------------------|-----------------|------------------------------------------------|
| id                 | str             | Unique resource identifier                     |
| name               | str             | Human-readable name                            |
| type               | str             | ec2 / ebs / s3 / rds                          |
| instance_size      | str or null     | EC2 size (t2.micro ... t2.2xlarge)             |
| cpu_utilization    | float           | CPU usage 0.0-100.0                            |
| cost_per_hour      | float           | Current hourly cost in USD                     |
| status             | str             | running / idle / terminated                    |
| critical           | bool            | Mission-critical — must not be terminated      |
| reserved           | bool            | Whether reserved pricing is active             |
| idle_hours         | int             | Hours since last active use                    |
| dependency_ids     | list[str]       | IDs of resources that depend on this one       |

---

## Reward Function

Rewards are dense — every step provides a meaningful signal.

| Component           | Value           | Trigger                                           |
|---------------------|-----------------|---------------------------------------------------|
| cost_reduction      | 0.0 – +0.50     | Any action that lowers hourly cost                |
| idle_bonus          | +0.10           | Terminating a cpu=0 resource with high idle_hours |
| strategy_bonus      | +0.10           | Reserving a high-cost, high-utilization instance  |
| waste_penalty       | -0.15           | Terminating an actively used non-critical resource|
| dependency_penalty  | -0.20 each      | Breaking a running resource's dependency          |
| critical_penalty    | -1.00           | Terminating a mission-critical resource           |
| noop_penalty        | -0.02           | Selecting the noop action                         |

All rewards are clipped to [-1.0, 1.0].

---

## Tasks

### Task 1 — Waste Cleanup (Easy)

Terminate all idle EBS volumes and S3 buckets (cpu_utilization == 0,
idle_hours > 0). Four wasteful resources exist among eight total. Critical
resources must not be terminated.

- Budget: $1.00/hr — Max steps: 12
- Grader: `score = (correctly_terminated / total_waste) - 0.25 * critical_terminated`
- Optimal score: 1.00

### Task 2 — Resource Optimization (Medium)

Resize over-provisioned EC2 instances (cpu_utilization < 20%) to at least
one smaller instance tier. Five of eight instances are over-provisioned.
Two are critical production servers that must not be terminated.

- Budget: $0.80/hr — Max steps: 16
- Grader: `score = (correctly_resized / overprovisioned_count) - 0.25 * critical_terminated`
- Optimal score: 1.00

### Task 3 — Strategic Planning (Hard)

Apply reserved pricing and terminate idle resources to bring total hourly
spend below $1.00/hr across a ten-resource mixed environment. Optimal
strategy requires combining multiple action types.

- Budget: $1.00/hr — Max steps: 20
- Grader: proportional to cost reduction toward budget target; full score
  only when under budget; penalty for critical terminations
- Optimal score: 1.00

All graders are fully deterministic. Same agent behavior always produces
the same score.

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
