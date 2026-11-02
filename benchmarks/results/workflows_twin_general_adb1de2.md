# Workflow benchmark: qwen3:8b, stack twin, adb1de2

| workflow | runner | passed | median s | failed checks | time.clock_s | time.idle_s |
|---|---|---|---|---|---|---|
| image_and_read_back | general | 1/1 | 64 | - | - | 0.0058 |
| image_at_field_of_view | general | 1/1 | 60 | - | 0.518 | 0.00529 |
| set_defocus | general | 1/1 | 35 | - | - | 0.00549 |
| move_and_image | general | 1/1 | 50 | - | 0 | 0.00614 |
| read_status_only | general | 1/1 | 32 | - | 0 | 0.0229 |
| blank_the_beam | general | 1/1 | 9 | - | - | 0.00754 |
| eds_spectrum_elements | general | 1/1 | 82 | - | 0.2 | 8.73 |
| compare_two_regions | general | 1/1 | 243 | - | 5.26 | 50.3 |
| describe_latest_image | general | 1/1 | 92 | - | - | - |
| count_tiles_in_run | general | 1/1 | 55 | - | - | - |
| short_exposure_spectrum | general | 1/1 | 38 | - | - | - |
| points_of_interest_on_map | general | 0/1 | 75 | answer_number_near | - | - |
| find_particle_then_eds | general | 1/1 | 79 | - | 0.718 | 1.31 |
| survey_rank_and_analyse | general | 1/1 | 214 | - | 43.4 | 55.3 |
| focus_series | general | 1/1 | 181 | - | 2.07 | 114 |
| map_300nm | general | 1/1 | 86 | - | 6.15 | 6.51 |
| map_600nm_with_eds | general | 1/1 | 147 | - | 40.7 | 7.28 |
| cuj2_map_with_eds_points | general | 1/1 | 89 | - | 22.9 | 7.55 |
| cuj3_three_regions_high_mag | general | 1/1 | 187 | - | 3.2 | 13.4 |
| map_1um_overview | general | 1/1 | 147 | - | 55.9 | 0.293 |
| map_and_stitch | general | 1/1 | 105 | - | 24.7 | 0.164 |
| explore_then_zoom | general | 1/1 | 151 | - | 24.7 | 8.97 |
| abort_then_resume | general | 1/1 | 829 | - | 503 | 29 |
| revisit_point_of_interest | general | 1/1 | 117 | - | 1.56 | 52.6 |
| zoom_to_atomic_columns | general | 1/1 | 166 | - | 1.04 | 0.0844 |
| held_move_is_approved | general | 1/1 | 80 | - | 1.02 | 39.2 |
| held_move_is_denied | general | 1/1 | 16 | - | - | 0.695 |
| held_map_is_approved | general | 1/1 | 123 | - | 6.15 | 39.6 |
| field_of_view_out_of_range | general | 1/1 | 46 | - | - | 0.0068 |
| dose_budget_refused | general | 1/1 | 171 | - | 0 | 164 |
| stage_out_of_range | general | 1/1 | 13 | - | - | 0.00683 |
| recover_from_unknown_detector | general | 1/1 | 70 | - | 1.04 | 42.7 |

Passed 31 of 32 runs; sleepwalking calls: 5.
