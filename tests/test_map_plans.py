"""Map plans the Mapper refuses or reshapes before anything moves, revisit-only plans, and the size of its reply.

Revisit-only plans image given stage points, or the best scored regions of a map made earlier, without
mapping again. The analysis server is pyTEMlib's real MapTools in this process (see test_map_pipeline).
"""

import importlib.util
import json

import pytest

from test_map_pipeline import PLAN, call
from test_map_pipeline_resilience import analysis, image_count, start, wait  # noqa: F401  (analysis is a fixture)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(importlib.util.find_spec("pyTEMlib.mcp_server.map_tools") is None, reason="needs the pyTEMlib fork with map tools"),
]


def refusal(mapper_proxy, plan):
    reply = call(mapper_proxy, "reserve", {"plan": plan})
    assert "error" in reply, reply
    if "reservation_id" in reply:
        call(mapper_proxy, "release", {"reservation_id": reply["reservation_id"]})
    assert reply["error"]["code"] == "plan_rejected", reply
    return reply["error"]["message"]


def run(mapper_proxy, plan):
    reserved, started = start(mapper_proxy, plan)
    assert wait(mapper_proxy, started["job_id"])["state"] == "done"
    results = call(mapper_proxy, "results", {"job_id": started["job_id"]})
    call(mapper_proxy, "release", {"reservation_id": reserved["reservation_id"]})
    return reserved, results


def test_an_unknown_field_is_named_with_the_field_meant_and_the_allowed_list(mapper_proxy):
    message = refusal(mapper_proxy, {**PLAN, "max_dose_e_per_a2": 20.0, "fov": 1e-7})
    assert "'max_dose_e_per_a2' (did you mean 'dose_budget_e_per_a2'?)" in message
    assert "'fov' (did you mean 'fov_m'?)" in message
    assert "Allowed fields: " in message and "dose_budget_e_per_a2" in message.split("Allowed fields: ")[1]


def test_the_example_plan_carries_the_callers_own_dose_limit_and_never_invents_one(mapper_proxy):
    with_dose = json.loads(refusal(mapper_proxy, {**PLAN, "max_dose_e_per_a2": 20.0}).split("Example plan: ", 1)[1])
    assert with_dose["dose_budget_e_per_a2"] == 20.0
    named = json.loads(refusal(mapper_proxy, {**PLAN, "label": "low dose", "colour": "red"}).split("Example plan: ", 1)[1])
    assert "dose_budget_e_per_a2" not in named
    without = json.loads(refusal(mapper_proxy, {**PLAN, "colour": "red"}).split("Example plan: ", 1)[1])
    assert "dose_budget_e_per_a2" not in without


def test_a_dose_refusal_names_only_the_remedies_for_what_adds_dose(mapper_proxy, analysis):
    tiles = refusal(mapper_proxy, {**PLAN, "dose_budget_e_per_a2": 1e-6})
    assert "lower dwell_s or overlap" in tiles and "revisit" not in tiles and "eds_exposure_s" not in tiles
    assert "give dose_budget_e_per_a2 only when they set a limit" in tiles
    revisit = refusal(mapper_proxy, {**PLAN, "find_regions": 1, "revisit": {"fov_m": 3e-8}, "dose_budget_e_per_a2": 1e-6})
    assert "widen revisit fov_m" in revisit


def test_a_plan_off_the_stage_is_refused_at_reserve_with_the_limit(mapper_proxy):
    message = refusal(mapper_proxy, {"box_m": [9.5e-4, 1.05e-3, 0.0, 2e-7], "fov_m": 1e-7})
    assert "outside the instrument's limits" in message and "asyncroscopy/stage/mapping/x" in message and "0.001]" in message
    message = refusal(mapper_proxy, {"revisit": {"fov_m": 3e-8, "points": [[0.0, 0.0], [0.0, -2e-3]]}})
    assert "revisit point:1 sets asyncroscopy/stage/mapping/y" in message and "[-0.001, 0.001]" in message
    reply = call(mapper_proxy, "reserve", {"plan": PLAN})
    assert reply["limits"]["targets"] == 4 and not reply["limits"]["unchecked"]
    call(mapper_proxy, "release", {"reservation_id": reply["reservation_id"]})


def test_given_points_are_imaged_without_mapping(mapper_proxy, mapping_twin_proxy, analysis):
    points = [[0.0, 0.0], [5e-8, -5e-8]]
    before = image_count(mapping_twin_proxy)
    reserved, results = run(mapper_proxy, {"revisit": {"fov_m": 3e-8, "points": points}})
    estimate = reserved["estimate"]
    assert estimate["tiles"] == 0 and estimate["phases"] == ["revisit"] and estimate["steps"] == 2
    summary = results["summary"]
    assert image_count(mapping_twin_proxy) - before == 2
    assert summary["revisits"]["imaged"] == 2 and "run_file" not in summary
    assert [c for v in summary["revisits"]["first"] for c in v["stage_xy_m"]] == pytest.approx([c for p in points for c in p])
    assert analysis.calls == []  # nothing to stitch or score
    assert json.loads(mapping_twin_proxy.get_parameters())["beam_blanked"] is True


def test_one_given_point_is_imaged_even_when_rounding_makes_its_box_a_hair_narrower_than_the_image(mapper_proxy, mapping_twin_proxy, analysis):
    point = [-2.021e-07, 3.52e-08]  # the box is point +/- 1e-8, and 3.52e-8 +/- 1e-8 is 1.9999999999999997e-8 wide
    _reserved, results = run(mapper_proxy, {"revisit": {"fov_m": 2e-8, "points": [point]}})
    assert results["summary"]["revisits"]["imaged"] == 1, results["summary"]


def test_one_given_point_is_not_refused_when_rounding_puts_it_a_hair_outside_its_own_box(mapper_proxy):
    # -4.43e-8 - 1e-7 + 1e-7 is -4.430000000000001e-08: the point was refused as outside its own box (qwen run 2026-10-31).
    reserved = call(mapper_proxy, "reserve", {"plan": {"revisit": {"fov_m": 2e-7, "points": [[-4.43e-8, 1.835e-7]]}}})
    assert "reservation_id" in reserved, reserved
    call(mapper_proxy, "release", {"reservation_id": reserved["reservation_id"]})


def test_a_revisit_of_given_points_without_a_field_of_view_keeps_the_current_one_and_a_maps_revisit_needs_one(mapper_proxy, mapping_twin_proxy, analysis):
    mapping_twin_proxy.set_fov(4e-8)
    reserved = call(mapper_proxy, "reserve", {"plan": {"revisit": {"points": [[0.0, 0.0]]}}})
    assert reserved["resolved_plan"]["revisit"]["fov_m"] == pytest.approx(4e-8), reserved
    call(mapper_proxy, "release", {"reservation_id": reserved["reservation_id"]})
    message = refusal(mapper_proxy, {**PLAN, "find_regions": 1, "revisit": {"imsize": 128}})
    assert "revisit fov_m is required with a map" in message


def test_given_points_stay_inside_the_box_and_the_dose_budget(mapper_proxy):
    message = refusal(mapper_proxy, {"box_m": [-1e-7, 1e-7, -1e-7, 1e-7], "revisit": {"fov_m": 3e-8, "points": [[9e-8, 0.0]]}})
    assert "outside the approved area" in message
    message = refusal(mapper_proxy, {"revisit": {"fov_m": 3e-8, "imsize": 512, "dwell_s": 1e-4, "points": [[0.0, 0.0]]}, "dose_budget_e_per_a2": 1.0})
    assert "above the budget" in message
    message = refusal(mapper_proxy, {"revisit": {"fov_m": 3e-8, "points": [[0.0, 0.0]], "map": "m"}})
    assert "not both" in message


def test_an_existing_maps_best_regions_are_imaged_without_mapping_again(mapper_proxy, mapping_twin_proxy, analysis):
    _, first = run(mapper_proxy, {**PLAN, "find_regions": 4})
    map_key = first["result_keys"]["map"]
    candidates = analysis.server.tools.list_pois(map_key, kind="candidate")["pois"]
    best = sorted(candidates, key=lambda p: p["score"], reverse=True)[:2]

    before = image_count(mapping_twin_proxy)
    reserved, results = run(mapper_proxy, {"revisit": {"fov_m": 3e-8, "map": map_key, "top_k": 2}})
    assert reserved["estimate"]["phases"] == ["select", "revisit"] and reserved["resolved_plan"]["box_m"] == pytest.approx(analysis.server.tools.map_info(map_key)["stage_box_m"])
    summary = results["summary"]
    assert [r["id"] for r in summary["regions"]["first"]] == [p["id"] for p in best]
    imaged = summary["revisits"]["imaged"]
    assert imaged + summary["revisits"]["skipped"] == 2 and image_count(mapping_twin_proxy) - before == imaged
    high_mag = analysis.server.tools.list_pois(map_key, kind="high_mag")["pois"]
    assert len(high_mag) == imaged and all(p["links"][1].startswith("poi:") for p in high_mag)


def test_a_map_scored_twice_has_each_region_imaged_once(mapper_proxy, mapping_twin_proxy, analysis):
    _, first = run(mapper_proxy, {**PLAN, "find_regions": 4})
    map_key = first["result_keys"]["map"]
    candidates = analysis.server.tools.list_pois(map_key, kind="candidate")["pois"]
    analysis.server.tools.store.add_pois(map_key, [{**p, "id": f"{p['id']}-again"} for p in candidates])
    best = sorted(candidates, key=lambda p: p["score"], reverse=True)[:2]

    _, results = run(mapper_proxy, {"revisit": {"fov_m": 3e-8, "map": map_key, "top_k": 2}})
    chosen = results["summary"]["regions"]["first"]
    assert [(r["x_m"], r["y_m"]) for r in chosen] == [(p["x_m"], p["y_m"]) for p in best]


def test_a_map_with_no_scored_regions_says_to_score_it(mapper_proxy, mapping_twin_proxy, analysis):
    _, first = run(mapper_proxy, PLAN)
    map_key = first["result_keys"]["map"]
    before = image_count(mapping_twin_proxy)
    _, results = run(mapper_proxy, {"revisit": {"fov_m": 3e-8, "map": map_key}})
    assert "score_map(map=" in results["summary"]["select_error"] and image_count(mapping_twin_proxy) == before


def test_the_done_reply_stays_short_however_many_tiles(mapper_proxy, analysis):
    small = {**PLAN, "per_tile": ["eds_point"], "eds_exposure_s": 0.01, "find_regions": 3, "revisit": {"fov_m": 3e-8}}
    _, results = run(mapper_proxy, small)
    summary = results["summary"]
    tiles = summary["grid"]["rows"] * summary["grid"]["cols"]
    assert summary["spectra"]["taken"] == tiles and len(summary["spectra"]["first_keys"]) == 3
    assert len(summary["regions"]["first"]) <= 3 and len(summary["revisits"]["first"]) <= 3
    assert "drift_fit" not in summary and set(summary["stitching"].get("drift", {})) <= {"total_nm", "mean_rate_nm_per_s"}
    _, bigger = run(mapper_proxy, {**small, "box_m": [-2e-7, 2e-7, -2e-7, 2e-7]})
    big = bigger["summary"]
    assert big["grid"]["rows"] * big["grid"]["cols"] > tiles
    assert len(json.dumps(big)) < 4000 and len(json.dumps(big)) - len(json.dumps(summary)) < 200


def test_revisit_fields_at_the_top_level_are_refused_with_the_plan_they_belong_in():
    from asyncroscopy.instruments.electron_microscope.mapper import check_plan_fields
    with pytest.raises(ValueError, match="belong inside revisit") as refused:
        check_plan_fields({"map": "map_r1_map", "top_k": 3, "fov_m": 6e-8})
    assert '{"revisit": {"map": "map_r1_map", "top_k": 3, "fov_m": 6e-08}}' in str(refused.value)
