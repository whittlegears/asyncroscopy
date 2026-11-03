# Workflow benchmark: qwen3:8b, stack twin, e299c88

| workflow | runner | passed | median s | failed checks | time.clock_s | time.idle_s |
|---|---|---|---|---|---|---|
| image_and_read_back | general | 1/1 | 68 | - | - | 0.00583 |
| image_at_field_of_view | general | 1/1 | 57 | - | 0.518 | 0.00586 |
| set_defocus | general | 1/1 | 36 | - | - | 0.00591 |
| move_and_image | general | 1/1 | 52 | - | - | 0.00534 |
| read_status_only | general | 1/1 | 28 | - | 0 | 0.0292 |
| blank_the_beam | general | 1/1 | 8 | - | - | 0.0076 |
| eds_spectrum_elements | general | 1/1 | 86 | - | 1 | 8.49 |
| compare_two_regions | general | 1/1 | 238 | - | 5.25 | 49.5 |
| describe_latest_image | general | 1/1 | 92 | - | - | - |
| count_tiles_in_run | general | 1/1 | 54 | - | - | - |
| short_exposure_spectrum | general | 1/1 | 58 | - | - | - |
| points_of_interest_on_map | general | 1/1 | 253 | - | - | - |
| find_particle_then_eds | general | 1/1 | 72 | - | 0.718 | 1.38 |
| survey_rank_and_analyse | general | 1/1 | 250 | - | 43.4 | 54 |
| focus_series | general | 1/1 | 177 | - | 2.07 | 109 |
| map_300nm | general | 1/1 | 84 | - | 6.15 | 6.28 |
| map_600nm_with_eds | general | 1/1 | 147 | - | 40.7 | 6.96 |
| cuj2_map_with_eds_points | general | 1/1 | 97 | - | 22.9 | 6.88 |
| cuj3_three_regions_high_mag | general | 1/1 | 182 | - | 3.24 | 12.5 |
| map_1um_overview | general | 1/1 | 135 | - | 55.9 | 0.242 |
| map_and_stitch | general | 1/1 | 68 | - | 24.7 | 0.156 |
| explore_then_zoom | general | 1/1 | 152 | - | 24.7 | 8.49 |
| abort_then_resume | general | 1/1 | 819 | - | 503 | 29 |
| revisit_point_of_interest | general | 1/1 | 148 | - | 1.57 | 27.6 |
| zoom_to_atomic_columns | general | 1/1 | 156 | - | 1.04 | 0.032 |
| held_move_is_approved | general | 1/1 | 78 | - | 1.03 | 38.4 |
| held_move_is_denied | general | 1/1 | 15 | - | - | 1.1 |
| held_map_is_approved | general | 1/1 | 117 | - | 6.14 | 38.4 |
| field_of_view_out_of_range | general | 1/1 | 44 | - | - | 0.00691 |
| dose_budget_refused | general | 1/1 | 127 | - | 0 | 103 |
| stage_out_of_range | general | 1/1 | 12 | - | - | 0.00697 |
| recover_from_unknown_detector | general | 1/1 | 66 | - | 1.04 | 42 |

Passed 32 of 32 runs; sleepwalking calls: 0.
