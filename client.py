"""
Python client for the Cloud FinOps Optimizer environment.

Extends openenv.core.env_client.EnvClient which handles all WebSocket and
HTTP communication. Implements the three required abstract methods:

    _step_payload  - serialize FinOpsAction to the JSON dict sent over the wire
    _parse_result  - deserialize wire response to StepResult
    _parse_state   - deserialize wire response to FinOpsState

All v2 fields (goal, last_action_error, dependency_graph, cascading_risks,
sla_violations, simulate_result, sla_max_cpu, sla_status, resize_cooldown_steps,
reservation_committed) are fully deserialized so the inference script can use them.

Usage (sync):

    from client import FinOpsEnv, FinOpsAction

    with FinOpsEnv(base_url="http://localhost:7860").sync() as env:
        result = env.reset(task_id="task1")
        result = env.step(FinOpsAction(action_type="terminate", resource_id="ebs-001",
                                       reasoning="idle 720 hrs, no dependents"))
        print(result.reward, result.done)
        print(result.observation.goal)
        print(result.observation.dependency_graph)

Usage (async):

    import asyncio
    from client import FinOpsEnv, FinOpsAction

    async def main():
        async with FinOpsEnv(base_url="http://localhost:7860") as env:
            result = await env.reset(task_id="task3")
            result = await env.step(FinOpsAction(
                action_type="simulate",
                simulate_action={"action_type": "reserve", "resource_id": "ec2-h01"},
                reasoning="checking reservation safety before committing"
            ))
            print(result.observation.simulate_result.recommendation)
    asyncio.run(main())
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

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


class FinOpsEnv(EnvClient[FinOpsAction, FinOpsObservation, FinOpsState]):
    """
    WebSocket/HTTP client for the Cloud FinOps Optimizer environment.

    The base class EnvClient handles connection management, WebSocket framing,
    and the sync() wrapper. You only need the three parsing methods below.
    """

    # ------------------------------------------------------------------
    # Required override 1: serialize action to wire format
    # ------------------------------------------------------------------

    def _step_payload(self, action: FinOpsAction) -> dict:
        """
        Serialize FinOpsAction to the JSON dict sent over the WebSocket.
        All fields must be included so the server receives simulate_action
        and reasoning correctly.
        """
        payload: Dict[str, Any] = {"action_type": action.action_type}
        if action.resource_id is not None:
            payload["resource_id"] = action.resource_id
        if action.target_size is not None:
            payload["target_size"] = action.target_size
        if action.simulate_action is not None:
            payload["simulate_action"] = action.simulate_action
        if action.reasoning is not None:
            payload["reasoning"] = action.reasoning
        return payload

    # ------------------------------------------------------------------
    # Required override 2: deserialize step/reset response
    # ------------------------------------------------------------------

    def _parse_result(self, payload: dict) -> StepResult:
        """
        Deserialize a step or reset response into a typed StepResult.

        The payload from /reset has the observation fields at the top level.
        The payload from /step wraps them under an "observation" key.
        Both shapes are handled via the obs_data fallback.
        """
        obs_data: dict = payload.get("observation") or payload

        resources = [
            self._parse_resource(r)
            for r in obs_data.get("resources", [])
        ]

        # Deserialize SimulateResult if present (Upgrade 5)
        sim_raw = obs_data.get("simulate_result")
        simulate_result: Optional[SimulateResult] = None
        if sim_raw:
            simulate_result = SimulateResult(
                proposed_action=sim_raw.get("proposed_action", {}),
                projected_cost_per_hour=sim_raw.get("projected_cost_per_hour", 0.0),
                projected_budget_remaining=sim_raw.get("projected_budget_remaining", 0.0),
                projected_reward=sim_raw.get("projected_reward", 0.0),
                projected_sla_violations=sim_raw.get("projected_sla_violations", []),
                cascading_risks=sim_raw.get("cascading_risks", []),
                recommendation=sim_raw.get("recommendation", ""),
                safe_to_apply=sim_raw.get("safe_to_apply", True),
            )

        observation = FinOpsObservation(
            done=payload.get("done", False),
            reward=payload.get("reward"),
            resources=resources,
            total_cost_per_hour=obs_data.get("total_cost_per_hour", 0.0),
            budget_per_hour=obs_data.get("budget_per_hour", 0.0),
            budget_remaining=obs_data.get("budget_remaining", 0.0),
            task_id=obs_data.get("task_id", ""),
            task_description=obs_data.get("task_description", ""),
            # v2 fields
            goal=obs_data.get("goal", ""),
            last_action_error=obs_data.get("last_action_error"),
            dependency_graph=obs_data.get("dependency_graph", {}),
            cascading_risks=obs_data.get("cascading_risks", {}),
            sla_violations=obs_data.get("sla_violations", []),
            simulate_result=simulate_result,
            step_count=obs_data.get("step_count", 0),
            max_steps=obs_data.get("max_steps", 20),
            info=obs_data.get("info", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    # ------------------------------------------------------------------
    # Required override 3: deserialize state response
    # ------------------------------------------------------------------

    def _parse_state(self, payload: dict) -> FinOpsState:
        """Deserialize a /state response into a typed FinOpsState."""
        return FinOpsState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            total_cost_per_hour=payload.get("total_cost_per_hour", 0.0),
            initial_cost_per_hour=payload.get("initial_cost_per_hour", 0.0),
            budget_per_hour=payload.get("budget_per_hour", 0.0),
            terminated_ids=payload.get("terminated_ids", []),
            reserved_ids=payload.get("reserved_ids", []),
            resize_history=payload.get("resize_history", {}),
            sla_violation_history=payload.get("sla_violation_history", []),
            reasoning_log=payload.get("reasoning_log", []),
            done=payload.get("done", False),
        )

    # ------------------------------------------------------------------
    # Helper: parse a single CloudResource dict from the wire
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_resource(r: dict) -> CloudResource:
        """
        Deserialize one resource dict including all v2 SLA and temporal fields.
        Defaults mirror the CloudResource model defaults so missing fields
        never cause KeyError or validation failure.
        """
        sla_status_raw = r.get("sla_status", "ok")
        try:
            sla_status = SLAStatus(sla_status_raw)
        except ValueError:
            sla_status = SLAStatus.OK

        return CloudResource(
            id=r["id"],
            name=r.get("name", ""),
            type=ResourceType(r["type"]),
            instance_size=InstanceSize(r["instance_size"]) if r.get("instance_size") else None,
            cpu_utilization=r.get("cpu_utilization", 0.0),
            memory_utilization=r.get("memory_utilization", 0.0),
            cost_per_hour=r.get("cost_per_hour", 0.0),
            status=ResourceStatus(r.get("status", "running")),
            critical=r.get("critical", False),
            reserved=r.get("reserved", False),
            idle_hours=r.get("idle_hours", 0),
            dependency_ids=r.get("dependency_ids", []),
            tags=r.get("tags", {}),
            # v2 SLA fields (Upgrade 2)
            sla_max_cpu=r.get("sla_max_cpu", 90.0),
            sla_uptime_pct=r.get("sla_uptime_pct", 99.9),
            sla_status=sla_status,
            # v2 temporal fields (Upgrade 3)
            resize_cooldown_steps=r.get("resize_cooldown_steps", 0),
            reservation_committed=r.get("reservation_committed", False),
        )
