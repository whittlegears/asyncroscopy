# Workflow benchmark: qwen3:8b, stack twin, 93e43c1

| workflow | runner | passed | median s | failed checks | time.clock_s | time.idle_s |
|---|---|---|---|---|---|---|
| image_and_read_back | general | 1/1 | 68 | - | - | 0.00762 |
| image_at_field_of_view | general | 0/1 | 96 | no_sleepwalking | 1.04 | 28.3 |
| set_defocus | general | 1/1 | 36 | - | - | 0.006 |
| move_and_image | general | 1/1 | 42 | - | 0 | 0.00523 |
| read_status_only | general | 1/1 | 28 | - | 0 | 0.0218 |
| blank_the_beam | general | 1/1 | 9 | - | - | 0.0165 |
| eds_spectrum_elements | general | 1/1 | 68 | - | 1 | 3.42 |
| compare_two_regions | general | 1/1 | 64 | - | - | 0.00525 |
| describe_latest_image | general | 1/1 | 95 | - | - | - |
| count_tiles_in_run | general | 1/1 | 56 | - | - | - |
| short_exposure_spectrum | general | 1/1 | 65 | - | - | - |
| points_of_interest_on_map | general | 1/1 | 339 | - | - | - |
| find_particle_then_eds | general | 1/1 | 71 | - | 0.718 | 1.32 |
| survey_rank_and_analyse | general | 1/1 | 285 | - | 43.3 | 55.3 |
| focus_series | general | 0/1 | 104 | called[1] | 1.04 | 46.7 |
| map_300nm | general | 1/1 | 88 | - | 6.15 | 6.75 |
| map_600nm_with_eds | general | 1/1 | 152 | - | 40.7 | 7.62 |
| cuj2_map_with_eds_points | general | 1/1 | 94 | - | 22.9 | 7.9 |
| cuj3_three_regions_high_mag | general | 1/1 | 198 | - | 3.18 | 13.3 |
| map_1um_overview | general | 1/1 | 143 | - | 55.9 | 0.26 |
| map_and_stitch | general | 1/1 | 109 | - | 24.7 | 0.15 |
| explore_then_zoom | general | 1/1 | 157 | - | 24.7 | 9.35 |
| abort_then_resume | general | 1/1 | 827 | - | 503 | 31.9 |
| revisit_point_of_interest | general | 1/1 | 172 | - | 0 | 0.119 |
| zoom_to_atomic_columns | general | 1/1 | 153 | - | 1.04 | 0.0301 |
| held_move_is_approved | general | 1/1 | 81 | - | 1.1 | 39.7 |
| held_move_is_denied | general | 1/1 | 18 | - | - | 0.208 |
| held_map_is_approved | general | 1/1 | 125 | - | 6.15 | 40.9 |
| field_of_view_out_of_range | general | 1/1 | 46 | - | - | 0.00626 |
| dose_budget_refused | general | 1/1 | 179 | - | 0 | 170 |
| stage_out_of_range | general | 1/1 | 14 | - | - | 0.00709 |
| recover_from_unknown_detector | general | 1/1 | 49 | - | 0.518 | 5.12 |

Passed 30 of 32 runs; sleepwalking calls: 7.
