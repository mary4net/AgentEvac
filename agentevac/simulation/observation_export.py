"""JSONL export for fixed-home environmental observations.

The exporter records one row per original spawn agent per decision round.  Rows
observe the agent's home edge even if the live AgentEvac vehicle has departed,
changed route, or arrived.
"""

import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from agentevac.agents.belief_model import update_agent_belief
from agentevac.agents.information_model import sample_environment_signal


SpawnEvent = Tuple[str, str, str, float, str, str, str, Any]
EdgeRiskFn = Callable[[str], Tuple[bool, float, float]]


def _round_or_none(value: Optional[float], digits: int = 2) -> Optional[float]:
    if value is None:
        return None
    if not math.isfinite(float(value)):
        return None
    return round(float(value), digits)


def distance_band_to_hazard(margin_m: Optional[float]) -> str:
    """Classify a fire-edge margin using the same bands as driver briefings."""
    if margin_m is None:
        return "unknown"
    margin = float(margin_m)
    if margin <= 0.0:
        return "inside_fire_zone"
    if margin <= 1200.0:
        return "very_close"
    if margin <= 2500.0:
        return "near"
    if margin <= 5000.0:
        return "buffered"
    return "clear"


class HomeObservationExporter:
    """Write fixed-home observation rows for downstream simulators."""

    def __init__(
        self,
        path: Optional[str],
        *,
        noise_enabled: bool,
        map_name: str,
        net_file: str,
        sigma_info: float,
        distance_ref_m: float,
        belief_inertia: float,
    ):
        self.enabled = bool(path)
        self.path = path
        self.noise_enabled = bool(noise_enabled)
        self.map_name = str(map_name)
        self.net_file = str(net_file)
        self.sigma_info = float(max(0.0, sigma_info))
        self.distance_ref_m = float(max(0.0, distance_ref_m))
        self.belief_inertia = float(belief_inertia)
        self._fh = None
        self._belief_by_agent: Dict[str, Dict[str, float]] = {}
        if self.enabled:
            target = Path(str(path))
            os.makedirs(target.parent or ".", exist_ok=True)
            self._fh = open(target, "w", encoding="utf-8")

    def close(self):
        if self._fh:
            self._fh.flush()
            self._fh.close()
            self._fh = None

    def write_round(
        self,
        *,
        tick: int,
        decision_round: int,
        sim_t_s: float,
        spawn_events: Iterable[SpawnEvent],
        fires: List[Dict[str, Any]],
        edge_risk: EdgeRiskFn,
    ) -> int:
        """Write one observation row for every original spawn event."""
        if not self.enabled or self._fh is None:
            return 0
        rows = 0
        for event in spawn_events:
            veh_id = str(event[0])
            home_edge = str(event[1])
            _, _, margin_m = edge_risk(home_edge)
            base_margin_m = _round_or_none(margin_m, 2)
            effective_sigma = self.sigma_info if self.noise_enabled else 0.0
            env_signal = sample_environment_signal(
                agent_id=veh_id,
                sim_t_s=sim_t_s,
                current_edge=home_edge,
                current_edge_margin_m=base_margin_m,
                route_head_min_margin_m=base_margin_m,
                decision_round=decision_round,
                sigma_info=effective_sigma,
                distance_ref_m=self.distance_ref_m,
            )
            env_signal["source_metric"] = "home_edge_margin_m"
            belief = update_agent_belief(
                prev_belief=self._belief_by_agent.get(veh_id, {}),
                env_signal=env_signal,
                social_signal={"message_count": 0},
                theta_trust=0.0,
                inertia=self.belief_inertia,
            )
            self._belief_by_agent[veh_id] = {
                "p_safe": float(belief["p_safe"]),
                "p_risky": float(belief["p_risky"]),
                "p_danger": float(belief["p_danger"]),
            }
            row = {
                "tick": int(tick),
                "decision_round": int(decision_round),
                "time_s": round(float(sim_t_s), 2),
                "agent_id": veh_id,
                "home_edge": home_edge,
                "current_edge": home_edge,
                "source_metric": "home_edge_margin_m",
                "base_margin_m": env_signal.get("base_margin_m"),
                "observed_margin_m": env_signal.get("observed_margin_m"),
                "observed_state": env_signal.get("observed_state"),
                "noise_delta_m": env_signal.get("noise_delta_m"),
                "sigma_info": env_signal.get("sigma_info"),
                "distance_band_to_hazard": distance_band_to_hazard(base_margin_m),
                "p_safe": round(float(belief["p_safe"]), 4),
                "p_risky": round(float(belief["p_risky"]), 4),
                "p_danger": round(float(belief["p_danger"]), 4),
                "uncertainty": round(float(belief["entropy_norm"]), 4),
                "fires": [
                    {
                        "id": str(fire.get("id")),
                        "x": float(fire.get("x", 0.0)),
                        "y": float(fire.get("y", 0.0)),
                        "r": round(float(fire.get("r", 0.0)), 2),
                    }
                    for fire in fires
                ],
                "map_name": self.map_name,
                "net_file": self.net_file,
            }
            self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows += 1
        self._fh.flush()
        return rows
