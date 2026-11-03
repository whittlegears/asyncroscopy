# Workflow benchmark: qwen3:8b, stack twin, 1ed13dc

| workflow | runner | passed | median s | failed checks | time.clock_s | time.idle_s |
|---|---|---|---|---|---|---|
| image_and_read_back | general | 1/1 | 93 | - | 0.518 | 36.3 |
| image_at_field_of_view | general | 1/1 | 55 | - | 0.518 | 0.00631 |
| set_defocus | general | 1/1 | 35 | - | - | 0.006 |
| move_and_image | general | 1/1 | 48 | - | - | 0.00465 |
| read_status_only | general | 1/1 | 26 | - | 0 | 0.0185 |
| blank_the_beam | general | 1/1 | 8 | - | - | 0.0047 |
| eds_spectrum_elements | general | 1/1 | 64 | - | 1 | 8.31 |
| compare_two_regions | general | 1/1 | 177 | - | 5.25 | 26.1 |
| describe_latest_image | general | 1/1 | 87 | - | - | - |
| count_tiles_in_run | general | 1/1 | 54 | - | - | - |
| short_exposure_spectrum | general | 1/1 | 37 | - | - | - |
| points_of_interest_on_map | general | 1/1 | 279 | - | - | - |
| find_particle_then_eds | general | 1/1 | 78 | - | 0.718 | 1.26 |
| survey_rank_and_analyse | general | 1/1 | 214 | - | 43.3 | 56.3 |
| focus_series | general | 0/1 | 134 | called[1] | 1.55 | 65.2 |
| map_300nm | general | 1/1 | 86 | - | 6.15 | 6.18 |
| map_600nm_with_eds | general | 1/1 | 143 | - | 40.7 | 6.95 |
| cuj2_map_with_eds_points | general | 1/1 | 99 | - | 22.9 | 6.84 |
| cuj3_three_regions_high_mag | general | 1/1 | 136 | - | 3.19 | 13.1 |
| map_1um_overview | general | 1/1 | 129 | - | 55.9 | 0.243 |
| map_and_stitch | general | 1/1 | 78 | - | 24.7 | 0.123 |
| explore_then_zoom | general | 1/1 | 141 | - | 24.7 | 8.57 |
| abort_then_resume | general | 1/1 | 820 | - | 503 | 31 |
| revisit_point_of_interest | general | 1/1 | 114 | - | 1.56 | 52.3 |
| zoom_to_atomic_columns | general | 1/1 | 140 | - | 1.04 | 0.088 |
| held_move_is_approved | general | 1/1 | 79 | - | 1.03 | 38.4 |
| held_move_is_denied | general | 1/1 | 15 | - | - | 1.08 |
| held_map_is_approved | general | 1/1 | 116 | - | 6.15 | 38.6 |
| field_of_view_out_of_range | general | 1/1 | 44 | - | - | 0.0051 |
| dose_budget_refused | general | 1/1 | 128 | - | 0 | 103 |
| stage_out_of_range | general | 1/1 | 13 | - | - | 0.00519 |
| recover_from_unknown_detector | general | 1/1 | 74 | - | 1.04 | 42.2 |

Passed 31 of 32 runs; sleepwalking calls: 0.
