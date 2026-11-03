# Workflow benchmark: qwen3:8b, stack twin, c751134

| workflow | runner | passed | median s | failed checks | time.clock_s | time.idle_s |
|---|---|---|---|---|---|---|
| image_and_read_back | general | 3/3 | 24 | - | - | 0.00532 |
| image_at_field_of_view | general | 3/3 | 27 | - | 0.518 | 0.00542 |
| set_defocus | general | 3/3 | 5 | - | - | 0.00589 |
| move_and_image | general | 3/3 | 34 | - | - | 0.00442 |
| read_status_only | general | 3/3 | 21 | - | 0 | 0.0299 |
| blank_the_beam | general | 3/3 | 4 | - | - | 0.00618 |
| eds_spectrum_elements | general | 3/3 | 42 | - | 2 | 10.6 |
| compare_two_regions | general | 3/3 | 123 | - | 3.21 | 0.00838 |
| describe_latest_image | general | 3/3 | 66 | - | - | - |
| count_tiles_in_run | general | 3/3 | 23 | - | - | - |
| short_exposure_spectrum | general | 3/3 | 42 | - | - | - |
| points_of_interest_on_map | general | 3/3 | 250 | - | - | - |
| find_particle_then_eds | general | 3/3 | 42 | - | 0.718 | 1.22 |
| survey_rank_and_analyse | general | 3/3 | 199 | - | 43.4 | 34.1 |
| focus_series | general | 3/3 | 135 | - | 1.55 | 99.6 |
| map_300nm | general | 3/3 | 48 | - | 6.15 | 6.09 |
| map_600nm_with_eds | general | 3/3 | 119 | - | 40.7 | 6.82 |
| cuj2_map_with_eds_points | general | 3/3 | 85 | - | 22.9 | 6.71 |
| cuj3_three_regions_high_mag | general | 3/3 | 156 | - | 3.12 | 13.4 |
| map_1um_overview | general | 3/3 | 132 | - | 55.9 | 0.249 |
| map_and_stitch | general | 3/3 | 78 | - | 24.7 | 0.139 |
| explore_then_zoom | general | 3/3 | 105 | - | 24.7 | 8.26 |
| abort_then_resume | general | 3/3 | 833 | - | 503 | 29.5 |
| revisit_point_of_interest | general | 3/3 | 94 | - | 1.56 | 14.2 |
| zoom_to_atomic_columns | general | 3/3 | 94 | - | 1.04 | 13.6 |
| held_move_is_approved | general | 3/3 | 20 | - | 1.03 | 9.52 |
| held_move_is_denied | general | 3/3 | 14 | - | - | 1.06 |
| held_map_is_approved | general | 3/3 | 64 | - | 6.15 | 19.3 |
| field_of_view_out_of_range | general | 3/3 | 11 | - | - | 0.00586 |
| dose_budget_refused | general | 3/3 | 103 | - | 0 | 78.7 |
| stage_out_of_range | general | 3/3 | 11 | - | - | 0.00581 |
| recover_from_unknown_detector | general | 3/3 | 30 | - | 0.518 | 3.62 |

Passed 96 of 96 runs; sleepwalking calls: 0.
