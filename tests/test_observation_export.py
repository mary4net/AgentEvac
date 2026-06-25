import json

from agentevac.simulation.observation_export import (
    HomeObservationExporter,
    distance_band_to_hazard,
)


def test_distance_band_to_hazard():
    assert distance_band_to_hazard(None) == "unknown"
    assert distance_band_to_hazard(-1.0) == "inside_fire_zone"
    assert distance_band_to_hazard(1200.0) == "very_close"
    assert distance_band_to_hazard(2500.0) == "near"
    assert distance_band_to_hazard(5000.0) == "buffered"
    assert distance_band_to_hazard(5000.1) == "clear"


def test_home_observation_export_writes_every_spawn_each_round(tmp_path):
    out = tmp_path / "home_observations.jsonl"
    exporter = HomeObservationExporter(
        str(out),
        noise_enabled=False,
        map_name="test_map",
        net_file="sumo/test.net.xml",
        sigma_info=40.0,
        distance_ref_m=500.0,
        belief_inertia=0.35,
    )
    spawns = [
        ("a1", "edge_a", "dest", 0.0, "first", "10", "max", (255, 0, 0, 255)),
        ("a2", "edge_b", "dest", 0.0, "first", "10", "max", (0, 0, 255, 255)),
    ]

    def edge_risk(edge_id):
        return (False, 0.5, 100.0 if edge_id == "edge_a" else 3000.0)

    rows_written = exporter.write_round(
        tick=10,
        decision_round=1,
        sim_t_s=2.0,
        spawn_events=spawns,
        fires=[{"id": "F0", "x": 1.0, "y": 2.0, "r": 30.061}],
        edge_risk=edge_risk,
    )
    exporter.close()

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert rows_written == 2
    assert [row["agent_id"] for row in rows] == ["a1", "a2"]
    assert rows[0]["home_edge"] == "edge_a"
    assert rows[0]["current_edge"] == "edge_a"
    assert rows[0]["source_metric"] == "home_edge_margin_m"
    assert rows[0]["base_margin_m"] == 100.0
    assert rows[0]["observed_margin_m"] == 100.0
    assert rows[0]["noise_delta_m"] == 0.0
    assert rows[0]["sigma_info"] == 0.0
    assert rows[0]["distance_band_to_hazard"] == "very_close"
    assert rows[0]["observed_state"] == "danger"
    assert rows[0]["p_danger"] == 0.75
    assert rows[0]["fires"] == [{"id": "F0", "x": 1.0, "y": 2.0, "r": 30.06}]
    assert rows[0]["map_name"] == "test_map"
    assert rows[0]["net_file"] == "sumo/test.net.xml"
    assert rows[1]["distance_band_to_hazard"] == "buffered"


def test_home_observation_export_keeps_belief_state_across_rounds(tmp_path):
    out = tmp_path / "home_observations.jsonl"
    exporter = HomeObservationExporter(
        str(out),
        noise_enabled=False,
        map_name="test_map",
        net_file="sumo/test.net.xml",
        sigma_info=40.0,
        distance_ref_m=500.0,
        belief_inertia=0.5,
    )
    spawns = [("a1", "edge_a", "dest", 0.0, "first", "10", "max", (255, 0, 0, 255))]
    margins = [100.0, 6000.0]

    def edge_risk(_edge_id):
        return (False, 0.5, margins.pop(0))

    exporter.write_round(
        tick=1,
        decision_round=1,
        sim_t_s=0.2,
        spawn_events=spawns,
        fires=[],
        edge_risk=edge_risk,
    )
    exporter.write_round(
        tick=2,
        decision_round=2,
        sim_t_s=60.2,
        spawn_events=spawns,
        fires=[],
        edge_risk=edge_risk,
    )
    exporter.close()

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["p_danger"] == 0.75
    assert rows[1]["base_margin_m"] == 6000.0
    assert rows[1]["p_danger"] == 0.4
