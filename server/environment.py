"""
Cloud FinOps Optimizer — Core environment logic.

Upgrades implemented in this version:
  Upgrade 1: Dependency Graph     — resources declare deps; terminating a dep
                                    cascades failure to dependents
  Upgrade 2: SLA Constraints      — each resource has sla_max_cpu; resizing a
                                    loaded resource can trigger SLA violation
  Upgrade 3: Temporal Effects     — resize takes 2 steps to stabilise;
                                    reservation is permanently committed
  Upgrade 4: Explainability Score — agent's reasoning field is scored in grade
  Upgrade 5: Simulate Mode        — "simulate" action previews outcome without
                                    mutating state

Three deterministic tasks:
  task1 (easy)   — Waste Cleanup
  task2 (medium) — Resource Optimization with SLA constraints
  task3 (hard)   — Strategic Planning with full dependency + SLA + temporal
"""

from __future__ import annotations

import copy
import uuid
from typing import Any, Dict, List, Optional, Tuple

from openenv.core.env_server import Environment

from models import (
    CloudResource,
    FinOpsAction,
    FinOpsObservation,
    FinOpsState,
    InstanceSize,
    ResourceStatus,
    ResourceType,
    SimulateResult,
    SLAStatus,
)


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

INSTANCE_COST: Dict[InstanceSize, float] = {
    InstanceSize.MICRO:      0.0116,
    InstanceSize.SMALL:      0.0232,
    InstanceSize.MEDIUM:     0.0464,
    InstanceSize.LARGE:      0.0928,
    InstanceSize.XLARGE:     0.1856,
    InstanceSize.TWO_XLARGE: 0.3712,
}

RESERVE_DISCOUNT     = 0.40
RESIZE_COOLDOWN_STEPS = 2     # steps until a resize fully stabilises (Upgrade 3)

SIZE_ORDER = [
    InstanceSize.MICRO, InstanceSize.SMALL, InstanceSize.MEDIUM,
    InstanceSize.LARGE, InstanceSize.XLARGE, InstanceSize.TWO_XLARGE,
]

# Approximate CPU headroom per instance tier (fraction of capacity available
# after typical workload migration; used for SLA projection in simulate).
SIZE_CPU_HEADROOM: Dict[InstanceSize, float] = {
    InstanceSize.MICRO:      0.10,
    InstanceSize.SMALL:      0.20,
    InstanceSize.MEDIUM:     0.30,
    InstanceSize.LARGE:      0.40,
    InstanceSize.XLARGE:     0.55,
    InstanceSize.TWO_XLARGE: 0.65,
}


def _size_index(size: InstanceSize) -> int:
    return SIZE_ORDER.index(size)


# ---------------------------------------------------------------------------
# Projected CPU after a resize (Upgrade 2 + 3)
# ---------------------------------------------------------------------------

def _projected_cpu_after_resize(resource: CloudResource, new_size: InstanceSize) -> float:
    """
    When an instance is resized smaller its workload stays the same but
    capacity decreases. We model this as:
        projected_cpu = current_cpu * (old_capacity / new_capacity)
    where capacity is proportional to (size_index + 1).
    """
    if resource.instance_size is None:
        return resource.cpu_utilization
    old_idx = _size_index(resource.instance_size) + 1
    new_idx = _size_index(new_size) + 1
    return min(100.0, resource.cpu_utilization * (old_idx / new_idx))


# ---------------------------------------------------------------------------
# Task resource definitions
# ---------------------------------------------------------------------------

def _task1_resources() -> List[CloudResource]:
    """
    8 resources; 4 idle waste, 4 active/critical.
    Dependency graph: ebs-002 -> rds-001 (prod db volume backs the DB).
    """
    return [
        CloudResource(id="ebs-001", name="old-backup-vol",   type=ResourceType.EBS,
                      cpu_utilization=0.0,  cost_per_hour=0.08,
                      status=ResourceStatus.IDLE,    critical=False, idle_hours=720,
                      sla_max_cpu=90.0),
        CloudResource(id="ebs-002", name="prod-db-vol",      type=ResourceType.EBS,
                      cpu_utilization=55.0, cost_per_hour=0.12,
                      status=ResourceStatus.RUNNING, critical=True, idle_hours=0,
                      dependency_ids=["rds-001"], sla_max_cpu=85.0),
        CloudResource(id="s3-001",  name="unused-archive",   type=ResourceType.S3,
                      cpu_utilization=0.0,  cost_per_hour=0.023,
                      status=ResourceStatus.IDLE,    critical=False, idle_hours=2160,
                      sla_max_cpu=90.0),
        CloudResource(id="s3-002",  name="active-assets",    type=ResourceType.S3,
                      cpu_utilization=10.0, cost_per_hour=0.023,
                      status=ResourceStatus.RUNNING, critical=True, idle_hours=0,
                      sla_max_cpu=90.0),
        CloudResource(id="ebs-003", name="test-scratch-vol", type=ResourceType.EBS,
                      cpu_utilization=0.0,  cost_per_hour=0.05,
                      status=ResourceStatus.IDLE,    critical=False, idle_hours=480,
                      sla_max_cpu=90.0),
        CloudResource(id="ec2-001", name="web-server-prod",  type=ResourceType.EC2,
                      instance_size=InstanceSize.MEDIUM,
                      cpu_utilization=72.0, cost_per_hour=INSTANCE_COST[InstanceSize.MEDIUM],
                      status=ResourceStatus.RUNNING, critical=True,  idle_hours=0,
                      sla_max_cpu=85.0),
        CloudResource(id="s3-003",  name="defunct-logs",     type=ResourceType.S3,
                      cpu_utilization=0.0,  cost_per_hour=0.015,
                      status=ResourceStatus.IDLE,    critical=False, idle_hours=1440,
                      sla_max_cpu=90.0),
        CloudResource(id="rds-001", name="prod-postgres",    type=ResourceType.RDS,
                      cpu_utilization=40.0, cost_per_hour=0.25,
                      status=ResourceStatus.RUNNING, critical=True,  idle_hours=0,
                      sla_max_cpu=80.0, sla_uptime_pct=99.99),
    ]


def _task2_resources() -> List[CloudResource]:
    """
    8 EC2 instances with SLA constraints.
    Dependency chain: batch-workers -> api-servers -> prod-apps.
    Over-provisioned: ec2-t01..t04, ec2-t08.
    Critical: ec2-t05, ec2-t06.
    Twist: ec2-t03/t04 have tight SLA — resizing them aggressively violates SLA.
    """
    return [
        CloudResource(id="ec2-t01", name="batch-worker-1",  type=ResourceType.EC2,
                      instance_size=InstanceSize.XLARGE,
                      cpu_utilization=8.0,  cost_per_hour=INSTANCE_COST[InstanceSize.XLARGE],
                      status=ResourceStatus.RUNNING, critical=False,
                      dependency_ids=["ec2-t03"], sla_max_cpu=90.0),
        CloudResource(id="ec2-t02", name="batch-worker-2",  type=ResourceType.EC2,
                      instance_size=InstanceSize.XLARGE,
                      cpu_utilization=9.0,  cost_per_hour=INSTANCE_COST[InstanceSize.XLARGE],
                      status=ResourceStatus.RUNNING, critical=False,
                      dependency_ids=["ec2-t04"], sla_max_cpu=90.0),
        CloudResource(id="ec2-t03", name="api-server-1",    type=ResourceType.EC2,
                      instance_size=InstanceSize.TWO_XLARGE,
                      cpu_utilization=15.0, cost_per_hour=INSTANCE_COST[InstanceSize.TWO_XLARGE],
                      status=ResourceStatus.RUNNING, critical=False,
                      dependency_ids=["ec2-t05"], sla_max_cpu=70.0),   # tight SLA
        CloudResource(id="ec2-t04", name="api-server-2",    type=ResourceType.EC2,
                      instance_size=InstanceSize.TWO_XLARGE,
                      cpu_utilization=12.0, cost_per_hour=INSTANCE_COST[InstanceSize.TWO_XLARGE],
                      status=ResourceStatus.RUNNING, critical=False,
                      dependency_ids=["ec2-t06"], sla_max_cpu=70.0),   # tight SLA
        CloudResource(id="ec2-t05", name="prod-app-1",      type=ResourceType.EC2,
                      instance_size=InstanceSize.LARGE,
                      cpu_utilization=65.0, cost_per_hour=INSTANCE_COST[InstanceSize.LARGE],
                      status=ResourceStatus.RUNNING, critical=True,
                      sla_max_cpu=85.0, sla_uptime_pct=99.9),
        CloudResource(id="ec2-t06", name="prod-app-2",      type=ResourceType.EC2,
                      instance_size=InstanceSize.LARGE,
                      cpu_utilization=68.0, cost_per_hour=INSTANCE_COST[InstanceSize.LARGE],
                      status=ResourceStatus.RUNNING, critical=True,
                      sla_max_cpu=85.0, sla_uptime_pct=99.9),
        CloudResource(id="ec2-t07", name="analytics",       type=ResourceType.EC2,
                      instance_size=InstanceSize.MEDIUM,
                      cpu_utilization=45.0, cost_per_hour=INSTANCE_COST[InstanceSize.MEDIUM],
                      status=ResourceStatus.RUNNING, critical=False,
                      sla_max_cpu=90.0),
        CloudResource(id="ec2-t08", name="dev-sandbox",     type=ResourceType.EC2,
                      instance_size=InstanceSize.LARGE,
                      cpu_utilization=5.0,  cost_per_hour=INSTANCE_COST[InstanceSize.LARGE],
                      status=ResourceStatus.RUNNING, critical=False,
                      sla_max_cpu=90.0),
    ]


def _task3_resources() -> List[CloudResource]:
    """
    10 resources with a full 3-tier dependency chain and tight SLAs.
    Dependency chain: frontend -> backend-api -> primary-db
                      ml-inference -> data-pipeline -> analytics-db
    Idle waste: ebs-h01, s3-h01 (safe to terminate)
    Strategy requires: terminate waste, reserve high-cost always-on instances,
    and must NOT violate SLA on the critical chain.
    """
    return [
        CloudResource(id="ec2-h01", name="web-frontend-1", type=ResourceType.EC2,
                      instance_size=InstanceSize.LARGE,
                      cpu_utilization=70.0, cost_per_hour=INSTANCE_COST[InstanceSize.LARGE],
                      status=ResourceStatus.RUNNING, critical=True,
                      dependency_ids=["ec2-h03"],
                      sla_max_cpu=85.0, sla_uptime_pct=99.9),
        CloudResource(id="ec2-h02", name="web-frontend-2", type=ResourceType.EC2,
                      instance_size=InstanceSize.LARGE,
                      cpu_utilization=68.0, cost_per_hour=INSTANCE_COST[InstanceSize.LARGE],
                      status=ResourceStatus.RUNNING, critical=True,
                      dependency_ids=["ec2-h04"],
                      sla_max_cpu=85.0, sla_uptime_pct=99.9),
        CloudResource(id="ec2-h03", name="backend-api-1",  type=ResourceType.EC2,
                      instance_size=InstanceSize.XLARGE,
                      cpu_utilization=55.0, cost_per_hour=INSTANCE_COST[InstanceSize.XLARGE],
                      status=ResourceStatus.RUNNING, critical=True,
                      dependency_ids=["rds-h01"],
                      sla_max_cpu=80.0, sla_uptime_pct=99.9),
        CloudResource(id="ec2-h04", name="backend-api-2",  type=ResourceType.EC2,
                      instance_size=InstanceSize.XLARGE,
                      cpu_utilization=52.0, cost_per_hour=INSTANCE_COST[InstanceSize.XLARGE],
                      status=ResourceStatus.RUNNING, critical=True,
                      dependency_ids=["rds-h01"],
                      sla_max_cpu=80.0, sla_uptime_pct=99.9),
        CloudResource(id="ec2-h05", name="ml-inference",   type=ResourceType.EC2,
                      instance_size=InstanceSize.TWO_XLARGE,
                      cpu_utilization=80.0, cost_per_hour=INSTANCE_COST[InstanceSize.TWO_XLARGE],
                      status=ResourceStatus.RUNNING, critical=False,
                      dependency_ids=["ec2-h06"],
                      sla_max_cpu=90.0),
        CloudResource(id="ec2-h06", name="data-pipeline",  type=ResourceType.EC2,
                      instance_size=InstanceSize.TWO_XLARGE,
                      cpu_utilization=78.0, cost_per_hour=INSTANCE_COST[InstanceSize.TWO_XLARGE],
                      status=ResourceStatus.RUNNING, critical=False,
                      dependency_ids=["rds-h02"],
                      sla_max_cpu=90.0),
        CloudResource(id="rds-h01", name="primary-db",     type=ResourceType.RDS,
                      cpu_utilization=45.0, cost_per_hour=0.40,
                      status=ResourceStatus.RUNNING, critical=True,
                      sla_max_cpu=75.0, sla_uptime_pct=99.99),
        CloudResource(id="rds-h02", name="analytics-db",   type=ResourceType.RDS,
                      cpu_utilization=30.0, cost_per_hour=0.25,
                      status=ResourceStatus.RUNNING, critical=False,
                      sla_max_cpu=90.0),
        CloudResource(id="ebs-h01", name="unused-snapshot", type=ResourceType.EBS,
                      cpu_utilization=0.0,  cost_per_hour=0.10,
                      status=ResourceStatus.IDLE,    critical=False, idle_hours=720,
                      sla_max_cpu=90.0),
        CloudResource(id="s3-h01",  name="cold-archive",   type=ResourceType.S3,
                      cpu_utilization=0.0,  cost_per_hour=0.015,
                      status=ResourceStatus.IDLE,    critical=False, idle_hours=2880,
                      sla_max_cpu=90.0),
    ]


TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "task1": {
        "description": (
            "Waste Cleanup: terminate all idle EBS volumes and S3 buckets "
            "(cpu_utilization == 0, idle_hours > 0). "
            "Do NOT terminate critical resources or break dependency chains."
        ),
        "resources_fn": _task1_resources,
        "budget_per_hour": 1.00,
        "max_steps": 12,
    },
    "task2": {
        "description": (
            "Resource Optimization: resize over-provisioned EC2 instances "
            "(cpu_utilization < 20%) down by at least one tier. "
            "Check SLA constraints before resizing — some instances have tight "
            "cpu caps (sla_max_cpu). Resizes take 2 steps to stabilise."
        ),
        "resources_fn": _task2_resources,
        "budget_per_hour": 0.80,
        "max_steps": 16,
    },
    "task3": {
        "description": (
            "Strategic Planning: use simulate to reason about actions, then "
            "terminate idle waste and reserve always-on instances to bring "
            "total hourly spend below $1.00/hr. Preserve the 3-tier dependency "
            "chain (frontend -> backend -> db) and respect SLA caps. "
            "Reservations are permanent commitments."
        ),
        "resources_fn": _task3_resources,
        "budget_per_hour": 1.00,
        "max_steps": 20,
    },
}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class CloudFinOpsEnvironment(Environment):
    """
    Cloud FinOps Optimizer — production-grade OpenEnv environment.

    Implements the OpenEnv Environment interface:
        reset(**kwargs)  -> FinOpsObservation
        step(action)     -> FinOpsObservation
        state            -> FinOpsState   (property)

    Additional method:
        grade()          -> Dict[str, Any]  — deterministic episode score
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        self._state = FinOpsState()
        self._resources: List[CloudResource] = []

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs) -> FinOpsObservation:
        # Extract task_id from kwargs if provided, otherwise default to task1
        task_id = kwargs.get("task_id", "task1")
        if task_id not in TASK_CONFIGS:
            raise ValueError(f"Unknown task_id '{task_id}'. Choose from: {list(TASK_CONFIGS)}")

        cfg = TASK_CONFIGS[task_id]
        self._resources = cfg["resources_fn"]()
        total_cost = self._active_cost()

        self._state = FinOpsState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_id=task_id,
            total_cost_per_hour=total_cost,
            initial_cost_per_hour=total_cost,
            budget_per_hour=cfg["budget_per_hour"],
            done=False,
        )
        return self._build_observation(reward=None, done=False)

    def step(self, action: FinOpsAction, **kwargs) -> FinOpsObservation:
        if not self._state.task_id:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        if self._state.done:
            return self._build_observation(
                reward=0.0, done=True,
                info={"message": "Episode already finished."},
                last_action_error="Episode already finished.",
            )

        # Upgrade 4: log agent reasoning
        if action.reasoning:
            self._state.reasoning_log.append(
                f"step{self._state.step_count}: {action.reasoning.strip()}"
            )

        # Upgrade 5: simulate mode — no state mutation
        if action.action_type == "simulate":
            sim_result = self._simulate(action.simulate_action or {})
            self._state.step_count += 1
            return self._build_observation(
                reward=0.0, done=False,
                simulate_result=sim_result,
            )

        # Upgrade 3: advance existing cooldown timers BEFORE applying the new action
        # so that a freshly-set cooldown is not decremented on the same step
        self._tick_cooldowns()

        reward, info = self._apply_action(action)
        self._state.step_count += 1
        self._state.total_cost_per_hour = self._active_cost()

        cfg  = TASK_CONFIGS[self._state.task_id]
        done = self._state.step_count >= cfg["max_steps"]
        self._state.done = done

        error_str = info.get("error") if isinstance(info, dict) else None
        return self._build_observation(
            reward=round(reward, 4),
            done=done,
            info=info,
            last_action_error=error_str,
        )

    @property
    def state(self) -> FinOpsState:
        return self._state

    # ------------------------------------------------------------------
    # Deterministic graders  (0.0 – 1.0)
    # ------------------------------------------------------------------

    def grade(self) -> Dict[str, Any]:
        task_id = self._state.task_id
        if not task_id:
            return {"score": 0.01, "reason": "No active episode."}
        
        if task_id == "task1":
            result = self._grade_task1()
        elif task_id == "task2":
            result = self._grade_task2()
        elif task_id == "task3":
            result = self._grade_task3()
        else:
            return {"score": 0.01, "reason": f"Unknown task {task_id}"}
            
        # Calculate bonus score
        expl_score = self._score_reasoning()
        result["explainability_score"] = expl_score
        
        # Phase 2 Strict Constraint: Score must be strictly between 0 and 1 (not 0 or 1).
        # We clamp the FINAL combined score here.
        final_score = result["score"] + (expl_score * 0.05)
        result["score"] = round(max(0.01, min(0.99, final_score)), 4)
        
        return result

    def _score_reasoning(self) -> float:
        """
        Advanced scoring logic for agent reasoning quality.
        Recognizes keyword coverage, logical structure, and contextual awareness.
        """
        log = self._state.reasoning_log
        if not log:
            return 0.1  # Base score for valid but empty history
            
        import re
        recent = log[-5:] # look at last 5 decisions
        
        keywords = {
            "cost": 0.1, "waste": 0.1, "idle": 0.1,
            "sla": 0.15, "breach": 0.15, "violation": 0.15,
            "dependency": 0.15, "cascade": 0.15, "chain": 0.15,
            "simulate": 0.15, "projected": 0.1, "safe": 0.1
        }
        
        hits = 0.0
        for text in recent:
            text_lower = text.lower()
            # 1. Keyword coverage
            for kw, val in keywords.items():
                if kw in text_lower:
                    hits += val
            
            # 2. Justification markers (e.g., "because", "due to")
            if re.search(r"(because|due|since|result|reason|why)", text_lower):
                hits += 0.2
            
            # 3. Contextual markers (e.g., resource IDs like 'ec2-001')
            if re.search(r"[a-z0-9]+-[a-z0-9]+", text_lower):
                hits += 0.2
                
            # 4. Strategy markers
            if "simulate" in text_lower and ("safe" in text_lower or "advice" in text_lower):
                hits += 0.2
        
        # Average hits per logged reasoning
        raw_score = 0.1 + (hits / len(recent))
        return round(max(0.1, min(0.99, raw_score)), 4)

    def _grade_task1(self) -> Dict[str, Any]:
        original     = _task1_resources()
        waste_ids    = {r.id for r in original if not r.critical and r.cpu_utilization == 0.0 and r.idle_hours > 0}
        critical_ids = {r.id for r in original if r.critical}
        by_id        = {r.id: r for r in self._resources}

        correctly_terminated = sum(1 for rid in waste_ids    if by_id[rid].status == ResourceStatus.TERMINATED)
        critical_terminated  = sum(1 for rid in critical_ids if by_id[rid].status == ResourceStatus.TERMINATED)
        sla_violations       = sum(1 for r in self._resources if r.sla_status == SLAStatus.VIOLATED)

        base  = correctly_terminated / len(waste_ids) if waste_ids else 1.0
        score = base - 0.25 * critical_terminated - 0.05 * sla_violations
        
        # Phase 2 Strict Constraint: strictly between 0 and 1
        score = max(0.01, min(0.99, score))
        
        return {
            "score":                round(score, 4),
            "waste_resources":      len(waste_ids),
            "correctly_terminated": correctly_terminated,
            "critical_terminated":  critical_terminated,
            "sla_violations":       sla_violations,
        }

    def _grade_task2(self) -> Dict[str, Any]:
        original        = _task2_resources()
        overprovisioned = {r.id for r in original if r.type == ResourceType.EC2 and r.cpu_utilization < 20.0}
        critical_ids    = {r.id for r in original if r.critical}
        orig_by_id      = {r.id: r for r in original}
        by_id           = {r.id: r for r in self._resources}

        correctly_resized = sum(
            1 for rid in overprovisioned
            if by_id[rid].status != ResourceStatus.TERMINATED
            and by_id[rid].instance_size is not None
            and orig_by_id[rid].instance_size is not None
            and _size_index(by_id[rid].instance_size) < _size_index(orig_by_id[rid].instance_size)
            and by_id[rid].sla_status != SLAStatus.VIOLATED   # SLA must hold
        )
        critical_terminated = sum(1 for rid in critical_ids if by_id[rid].status == ResourceStatus.TERMINATED)
        sla_violations      = sum(1 for r in self._resources if r.sla_status == SLAStatus.VIOLATED)

        base  = correctly_resized / len(overprovisioned) if overprovisioned else 1.0
        score = base - 0.25 * critical_terminated - 0.05 * sla_violations
        
        # Phase 2 Strict Constraint: strictly between 0 and 1
        score = max(0.01, min(0.99, score))
        
        return {
            "score":                    round(score, 4),
            "overprovisioned_resources": len(overprovisioned),
            "correctly_resized":        correctly_resized,
            "critical_terminated":      critical_terminated,
            "sla_violations":           sla_violations,
        }

    def _grade_task3(self) -> Dict[str, Any]:
        original     = _task3_resources()
        critical_ids = {r.id for r in original if r.critical}
        by_id        = {r.id: r for r in self._resources}

        current_cost         = self._active_cost()
        budget               = self._state.budget_per_hour
        initial              = self._state.initial_cost_per_hour
        critical_terminated  = sum(1 for rid in critical_ids if by_id[rid].status == ResourceStatus.TERMINATED)
        sla_violations       = sum(1 for r in self._resources if r.sla_status == SLAStatus.VIOLATED)

        if current_cost <= budget:
            base = 1.0
        else:
            needed = initial - budget
            base   = min(1.0, max(0.0, (initial - current_cost) / needed)) if needed > 0 else 1.0

        score = base - 0.30 * critical_terminated - 0.05 * sla_violations
        
        # Phase 2 Strict Constraint: strictly between 0 and 1
        score = max(0.01, min(0.99, score))
        
        return {
            "score":                  round(score, 4),
            "initial_cost_per_hour":  round(initial, 4),
            "current_cost_per_hour":  round(current_cost, 4),
            "budget_per_hour":        budget,
            "under_budget":           current_cost <= budget,
            "critical_terminated":    critical_terminated,
            "sla_violations":         sla_violations,
        }

    # ------------------------------------------------------------------
    # Upgrade 5: Simulate (no state mutation)
    # ------------------------------------------------------------------

    def _simulate(self, proposed: Dict[str, Any]) -> SimulateResult:
        """
        Project the outcome of a proposed action without changing any state.
        Returns a SimulateResult the agent can inspect before committing.
        """
        action_type = proposed.get("action_type", "noop")
        resource_id = proposed.get("resource_id")
        target_size = proposed.get("target_size")

        projected_cost    = self._active_cost()
        projected_reward  = 0.0
        sla_violations:   List[str] = []
        cascade_risks:    List[str] = []
        safe              = True
        recommendation    = "Action appears safe."

        r = self._find(resource_id) if resource_id else None

        if action_type == "terminate" and r is not None:
            # Always compute cascade risks (useful even for critical resources)
            for other in self._resources:
                if r.id in (other.dependency_ids or []) and other.status == ResourceStatus.RUNNING:
                    cascade_risks.append(other.id)

            if r.critical:
                projected_reward = -1.0
                safe             = False
                cascade_note     = f" Also cascades to: {cascade_risks}." if cascade_risks else ""
                recommendation   = (
                    f"UNSAFE: '{r.id}' is critical. Terminating it causes a -1.0 penalty.{cascade_note}"
                )
            elif cascade_risks:
                projected_cost -= r.cost_per_hour
                safe           = False
                recommendation = (
                    f"WARNING: terminating '{r.id}' will cascade to: {cascade_risks}. "
                    "Penalty -0.20 per affected resource."
                )
            else:
                projected_cost -= r.cost_per_hour
                idle_bonus = 0.10 if r.cpu_utilization == 0.0 and r.idle_hours > 100 else 0.0
                projected_reward = min(0.5, r.cost_per_hour * 2.0) + idle_bonus
                recommendation   = (
                    f"SAFE: terminating '{r.id}' saves ${r.cost_per_hour:.4f}/hr. "
                    f"Projected reward: {projected_reward:+.4f}."
                )

        elif action_type == "resize" and r is not None and target_size is not None:
            try:
                new_size      = InstanceSize(target_size)
                proj_cpu      = _projected_cpu_after_resize(r, new_size)
                cost_saving   = r.cost_per_hour - INSTANCE_COST[new_size]
                projected_cost -= cost_saving
                if proj_cpu > r.sla_max_cpu:
                    sla_violations.append(r.id)
                    safe           = False
                    recommendation = (
                        f"UNSAFE: resizing '{r.id}' to {new_size} projects CPU at "
                        f"{proj_cpu:.1f}% (SLA cap {r.sla_max_cpu:.1f}%). "
                        f"Consider a less aggressive resize."
                    )
                else:
                    projected_reward = min(0.5, cost_saving * 3.0)
                    if r.cpu_utilization < 20.0:
                        projected_reward += 0.10
                    recommendation = (
                        f"SAFE: resize '{r.id}' to {new_size} saves ${cost_saving:.4f}/hr. "
                        f"Projected CPU: {proj_cpu:.1f}% (cap {r.sla_max_cpu:.1f}%). "
                        f"Reward: {projected_reward:+.4f}. "
                        f"Note: cooldown {RESIZE_COOLDOWN_STEPS} steps."
                    )
            except ValueError:
                safe           = False
                recommendation = f"INVALID: '{target_size}' is not a valid instance size."

        elif action_type == "reserve" and r is not None:
            if r.reserved:
                projected_reward = -0.02
                recommendation   = f"'{r.id}' is already reserved. No effect."
            else:
                saving           = r.cost_per_hour * RESERVE_DISCOUNT
                projected_cost  -= saving
                projected_reward = min(0.5, saving * 2.5)
                if r.cpu_utilization > 50.0 and r.cost_per_hour > 0.10:
                    projected_reward += 0.10
                recommendation = (
                    f"SAFE: reserving '{r.id}' saves ${saving:.4f}/hr (40% discount). "
                    f"Reward: {projected_reward:+.4f}. "
                    f"NOTE: reservation is a permanent commitment."
                )

        elif action_type == "noop":
            recommendation = "Noop applies a -0.02 penalty. Only use if no better action exists."

        return SimulateResult(
            proposed_action=proposed,
            projected_cost_per_hour=round(projected_cost, 6),
            projected_budget_remaining=round(self._state.budget_per_hour - projected_cost, 6),
            projected_reward=round(projected_reward, 4),
            projected_sla_violations=sla_violations,
            cascading_risks=cascade_risks,
            recommendation=recommendation,
            safe_to_apply=safe,
        )

    # ------------------------------------------------------------------
    # Upgrade 3: Cooldown ticker
    # ------------------------------------------------------------------

    def _tick_cooldowns(self) -> None:
        """
        Advance resize cooldown timers; mark SLA status accordingly.
        Only updates sla_status to AT_RISK if the resource is not already VIOLATED.
        Expired cooldowns trigger a full SLA re-evaluation.
        """
        for r in self._resources:
            if r.resize_cooldown_steps > 0:
                r.resize_cooldown_steps -= 1
                if r.resize_cooldown_steps > 0:
                    # Still in cooldown — mark at-risk only if not already violated
                    if r.sla_status != SLAStatus.VIOLATED:
                        r.sla_status = SLAStatus.AT_RISK
                else:
                    # Cooldown expired — re-evaluate SLA from actual cpu_utilization
                    self._update_sla_status(r)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _active_cost(self) -> float:
        return round(
            sum(r.cost_per_hour for r in self._resources if r.status != ResourceStatus.TERMINATED),
            6,
        )

    def _find(self, resource_id: Optional[str]) -> Optional[CloudResource]:
        if resource_id is None:
            return None
        for r in self._resources:
            if r.id == resource_id:
                return r
        return None

    def _update_sla_status(self, r: CloudResource) -> None:
        if r.cpu_utilization > r.sla_max_cpu:
            r.sla_status = SLAStatus.VIOLATED
            if r.id not in self._state.sla_violation_history:
                self._state.sla_violation_history.append(r.id)
        elif r.cpu_utilization > r.sla_max_cpu * 0.85:
            r.sla_status = SLAStatus.AT_RISK
        else:
            r.sla_status = SLAStatus.OK

    def _build_dependency_graph(self) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """
        Returns:
            dep_graph     : resource_id -> list of its dependency_ids (what it needs)
            cascade_risks : resource_id -> list of resources that depend on it
        """
        dep_graph:     Dict[str, List[str]] = {}
        cascade_risks: Dict[str, List[str]] = {}
        active = [r for r in self._resources if r.status != ResourceStatus.TERMINATED]
        for r in active:
            dep_graph[r.id] = [d for d in r.dependency_ids
                                if any(x.id == d and x.status != ResourceStatus.TERMINATED
                                       for x in active)]
        for r in active:
            for dep_id in r.dependency_ids:
                cascade_risks.setdefault(dep_id, [])
                if r.id not in cascade_risks[dep_id]:
                    cascade_risks[dep_id].append(r.id)
        return dep_graph, cascade_risks

    def _active_sla_violations(self) -> List[str]:
        return [r.id for r in self._resources if r.sla_status == SLAStatus.VIOLATED]

    def _build_observation(
        self,
        reward: Optional[float],
        done: bool,
        info: Optional[Dict[str, Any]] = None,
        last_action_error: Optional[str] = None,
        simulate_result: Optional[SimulateResult] = None,
    ) -> FinOpsObservation:
        cfg  = TASK_CONFIGS.get(self._state.task_id, {})
        cost = self._active_cost()
        desc = cfg.get("description", "")
        dep_graph, cascade_risks = self._build_dependency_graph()
        
        info = info or {
            "episode_id":       self._state.episode_id,
            "terminated_count": len(self._state.terminated_ids),
            "reserved_count":   len(self._state.reserved_ids),
            "resize_count":     len(self._state.resize_history),
            "sla_violations":   self._state.sla_violation_history,
        }
        info["finops_metrics"] = {
            "active_waste_count": sum(1 for r in self._resources if not r.critical and r.cpu_utilization == 0.0 and r.status != ResourceStatus.TERMINATED),
            "at_risk_count":      sum(1 for r in self._resources if r.sla_status == SLAStatus.AT_RISK),
            "violation_count":    sum(1 for r in self._resources if r.sla_status == SLAStatus.VIOLATED),
            "total_savings_potential_hr": sum(r.cost_per_hour for r in self._resources if not r.critical and r.cpu_utilization < 20.0 and r.status != ResourceStatus.TERMINATED),
        }

        return FinOpsObservation(
            done=done,
            reward=reward,
            resources=self._resources,
            total_cost_per_hour=cost,
            budget_per_hour=self._state.budget_per_hour,
            budget_remaining=round(self._state.budget_per_hour - cost, 6),
            task_id=self._state.task_id,
            task_description=desc,
            goal=(
                f"[{self._state.task_id}] Reduce cloud costs below "
                f"${self._state.budget_per_hour:.2f}/hr. "
                f"{desc.split(chr(46))[0]}."
            ),
            last_action_error=last_action_error,
            dependency_graph=dep_graph,
            cascading_risks=cascade_risks,
            sla_violations=self._active_sla_violations(),
            simulate_result=simulate_result,
            step_count=self._state.step_count,
            max_steps=cfg.get("max_steps", 20),
            info=info or {
                "episode_id":       self._state.episode_id,
                "terminated_count": len(self._state.terminated_ids),
                "reserved_count":   len(self._state.reserved_ids),
                "resize_count":     len(self._state.resize_history),
                "sla_violations":   self._state.sla_violation_history,
            },
        )

    # ------------------------------------------------------------------
    # Action dispatch
    # ------------------------------------------------------------------

    def _apply_action(self, action: FinOpsAction) -> Tuple[float, Dict[str, Any]]:
        t = action.action_type
        if t == "noop":
            return -0.02, {"explanation": "No action taken."}
        if t == "terminate":
            return self._terminate(action.resource_id)
        if t == "resize":
            return self._resize(action.resource_id, action.target_size)
        if t == "reserve":
            return self._reserve(action.resource_id)
        return -0.05, {"error": f"Unknown action_type '{t}'."}

    def _terminate(self, resource_id: Optional[str]) -> Tuple[float, Dict[str, Any]]:
        r = self._find(resource_id)
        if r is None:
            return -0.05, {"error": f"Resource '{resource_id}' not found."}
        if r.status == ResourceStatus.TERMINATED:
            return -0.03, {"explanation": "Already terminated."}
        if r.critical:
            r.status = ResourceStatus.TERMINATED
            self._state.terminated_ids.append(r.id)
            return -1.0, {"explanation": f"CRITICAL VIOLATION: terminated '{r.id}'."}

        # Upgrade 1: cascade penalty
        dep_penalty = 0.0
        affected    = []
        for other in self._resources:
            if r.id in other.dependency_ids and other.status == ResourceStatus.RUNNING:
                dep_penalty -= 0.20
                affected.append(other.id)

        old_cost = r.cost_per_hour
        r.status = ResourceStatus.TERMINATED
        self._state.terminated_ids.append(r.id)

        cost_reward   = min(0.5, old_cost * 2.0)
        idle_bonus    =  0.10 if r.cpu_utilization == 0.0 and r.idle_hours > 100 else 0.0
        waste_penalty = -0.15 if r.cpu_utilization > 30.0 else 0.0
        total = max(-1.0, min(1.0, cost_reward + idle_bonus + waste_penalty + dep_penalty))
        info  = {"explanation": f"Terminated '{r.id}'."}
        if affected:
            info["cascade_affected"] = affected
        return total, info

    def _resize(self, resource_id: Optional[str], target_size: Optional[str]) -> Tuple[float, Dict[str, Any]]:
        r = self._find(resource_id)
        if r is None:
            return -0.05, {"error": f"Resource '{resource_id}' not found."}
        if r.status == ResourceStatus.TERMINATED:
            return -0.03, {"explanation": "Cannot resize a terminated resource."}
        if r.type != ResourceType.EC2 or r.instance_size is None:
            return -0.05, {"explanation": "Resize only valid for EC2 instances."}
        try:
            new_size = InstanceSize(target_size)
        except ValueError:
            return -0.05, {"error": f"Invalid target_size '{target_size}'."}
        if _size_index(new_size) >= _size_index(r.instance_size):
            return -0.05, {"explanation": "target_size must be smaller than current size."}

        old_cost  = r.cost_per_hour
        old_size  = r.instance_size          # capture BEFORE mutation for history
        old_cpu   = r.cpu_utilization        # capture BEFORE mutation for reward calc
        proj_cpu  = _projected_cpu_after_resize(r, new_size)

        # Upgrade 2: SLA check against projected CPU
        sla_penalty = 0.0
        if proj_cpu > r.sla_max_cpu:
            r.sla_status = SLAStatus.VIOLATED
            sla_penalty  = -0.20
            if r.id not in self._state.sla_violation_history:
                self._state.sla_violation_history.append(r.id)

        # Upgrade 3: apply mutation — cooldown, new size, projected cpu
        r.instance_size          = new_size
        r.cost_per_hour          = INSTANCE_COST[new_size]
        r.cpu_utilization        = proj_cpu
        r.resize_cooldown_steps  = RESIZE_COOLDOWN_STEPS
        if r.sla_status != SLAStatus.VIOLATED:
            r.sla_status = SLAStatus.AT_RISK  # uncertain until cooldown expires
        # Record actual old_size -> new_size (old_size captured before mutation)
        self._state.resize_history[r.id] = f"{old_size.value}->{new_size.value}"

        cost_reward           = min(0.5, (old_cost - r.cost_per_hour) * 3.0)
        overprovisioned_bonus =  0.10 if old_cpu < 20.0 else 0.0   # use pre-resize cpu
        risk_penalty          = -0.10 if proj_cpu > 70.0 else 0.0  # use post-resize cpu
        total = max(-1.0, min(1.0, cost_reward + overprovisioned_bonus + risk_penalty + sla_penalty))
        return total, {
            "explanation": f"Resized '{r.id}' to {new_size}. Projected CPU: {proj_cpu:.1f}%.",
            "cooldown_steps": RESIZE_COOLDOWN_STEPS,
            "sla_status": r.sla_status.value,
        }

    def _reserve(self, resource_id: Optional[str]) -> Tuple[float, Dict[str, Any]]:
        r = self._find(resource_id)
        if r is None:
            return -0.05, {"error": f"Resource '{resource_id}' not found."}
        if r.status == ResourceStatus.TERMINATED:
            return -0.03, {"explanation": "Cannot reserve a terminated resource."}
        if r.reserved:
            return -0.02, {"explanation": "Resource already reserved."}

        old_cost = r.cost_per_hour
        r.reserved               = True
        r.reservation_committed  = True   # Upgrade 3: permanent
        r.cost_per_hour          = round(old_cost * (1.0 - RESERVE_DISCOUNT), 6)
        self._state.reserved_ids.append(r.id)

        cost_reward    = min(0.5, (old_cost - r.cost_per_hour) * 2.5)
        strategy_bonus = 0.10 if r.cpu_utilization > 50.0 and old_cost > 0.10 else 0.0
        total = max(-1.0, min(1.0, cost_reward + strategy_bonus))
        return total, {
            "explanation": f"Reserved '{r.id}' at {int(RESERVE_DISCOUNT*100)}% discount. Permanent commitment.",
        }
