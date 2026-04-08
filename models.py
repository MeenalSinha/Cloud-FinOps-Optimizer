"""
Typed Pydantic models for the Cloud FinOps Optimizer environment.

Action, Observation, and State inherit from the openenv.core.env_server base
classes as specified in the OpenEnv framework. These classes are Pydantic
BaseModel subclasses — fields are declared directly as class attributes.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from openenv.core.env_server import Action, Observation, State


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ResourceType(str, Enum):
    EC2     = "ec2"
    EBS     = "ebs"
    S3      = "s3"
    RDS     = "rds"


class ResourceStatus(str, Enum):
    RUNNING    = "running"
    IDLE       = "idle"
    TERMINATED = "terminated"
    DEGRADED   = "degraded"   # SLA violation — still running but impaired


class InstanceSize(str, Enum):
    MICRO      = "t2.micro"
    SMALL      = "t2.small"
    MEDIUM     = "t2.medium"
    LARGE      = "t2.large"
    XLARGE     = "t2.xlarge"
    TWO_XLARGE = "t2.2xlarge"


class SLAStatus(str, Enum):
    OK       = "ok"
    AT_RISK  = "at_risk"    # resize brought utilization close to limit
    VIOLATED = "violated"   # utilization > SLA threshold post-action


# ---------------------------------------------------------------------------
# Resource sub-model
# ---------------------------------------------------------------------------

class CloudResource(BaseModel):
    id: str
    name: str
    type: ResourceType
    instance_size: Optional[InstanceSize] = None
    cpu_utilization: float = Field(ge=0.0, le=100.0)
    memory_utilization: float = Field(ge=0.0, le=100.0, default=0.0)
    cost_per_hour: float = Field(ge=0.0)
    status: ResourceStatus
    critical: bool = False
    reserved: bool = False
    idle_hours: int = Field(ge=0, default=0)
    # Dependency graph: ids of resources that THIS resource depends on.
    # Terminating a dependency cascades failure to this resource.
    dependency_ids: List[str] = Field(default_factory=list)
    tags: Dict[str, str] = Field(default_factory=dict)
    # SLA fields — added for Upgrade 2 (Performance Constraints)
    sla_max_cpu: float = Field(ge=0.0, le=100.0, default=90.0)   # cpu breach threshold
    sla_uptime_pct: float = Field(ge=0.0, le=100.0, default=99.9) # required uptime
    sla_status: SLAStatus = SLAStatus.OK
    # Temporal effect fields — added for Upgrade 3 (Action Consequences Over Time)
    # resize_cooldown_steps: steps remaining before a resize takes full effect.
    # During cooldown the resource runs at elevated risk.
    resize_cooldown_steps: int = Field(ge=0, default=0)
    # reservation_committed: True once reserved — cannot be unreserved.
    reservation_committed: bool = False


# ---------------------------------------------------------------------------
# Simulate result — returned by the "simulate" action (Upgrade 5)
# ---------------------------------------------------------------------------

class SimulateResult(BaseModel):
    """
    Projected outcome of a proposed action, computed without mutating state.
    The agent can call simulate before committing to act.
    """
    proposed_action: Dict[str, Any]
    projected_cost_per_hour: float
    projected_budget_remaining: float
    projected_reward: float
    projected_sla_violations: List[str]          # resource ids that would violate SLA
    cascading_risks: List[str]                   # resource ids at cascade risk
    recommendation: str                          # plain-English recommendation
    safe_to_apply: bool                          # True if no SLA or cascade risk


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class FinOpsAction(Action):
    """
    One action the agent can take in the cloud environment.

    action_type choices:
        terminate  - delete resource_id (idle/unused resources only)
        resize     - downsize resource_id EC2 instance to target_size
        reserve    - apply 40% reserved-instance discount to resource_id
        simulate   - project outcome of a proposed sub-action without applying it
        noop       - take no action
    """
    action_type: str = "noop"
    resource_id: Optional[str] = None
    target_size: Optional[str] = None
    # For action_type == "simulate": the sub-action to preview
    simulate_action: Optional[Dict[str, Any]] = None
    # Reasoning field — agent can explain its decision (Upgrade 4)
    reasoning: Optional[str] = None


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class FinOpsObservation(Observation):
    """
    What the agent observes after each step.

    Inherits from Observation:
        done:   bool
        reward: Optional[float]
    """
    resources: List[CloudResource] = Field(default_factory=list)
    total_cost_per_hour: float = 0.0
    budget_per_hour: float = 0.0
    budget_remaining: float = 0.0
    task_id: str = ""
    task_description: str = ""
    goal: str = ""
    step_count: int = 0
    max_steps: int = 20
    last_action_error: Optional[str] = None
    # Dependency graph exposed to agent (Upgrade 1)
    dependency_graph: Dict[str, List[str]] = Field(default_factory=dict)
    # Resources currently at cascade risk if a dependency is removed (Upgrade 1)
    cascading_risks: Dict[str, List[str]] = Field(default_factory=dict)
    # SLA violations active this step (Upgrade 2)
    sla_violations: List[str] = Field(default_factory=list)
    # Simulate result if last action was a simulation (Upgrade 5)
    simulate_result: Optional[SimulateResult] = None
    info: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class FinOpsState(State):
    """
    Internal episode snapshot.

    Inherits from State:
        episode_id: Optional[str]
        step_count: int
    """
    task_id: str = ""
    total_cost_per_hour: float = 0.0
    initial_cost_per_hour: float = 0.0
    budget_per_hour: float = 0.0
    terminated_ids: List[str] = Field(default_factory=list)
    reserved_ids: List[str] = Field(default_factory=list)
    resize_history: Dict[str, str] = Field(default_factory=dict)
    sla_violation_history: List[str] = Field(default_factory=list)  # ids that ever violated
    reasoning_log: List[str] = Field(default_factory=list)          # agent reasoning strings
    done: bool = False
