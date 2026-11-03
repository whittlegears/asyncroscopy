# Workflow benchmark: qwen3:8b, stack twin, 1593771

| workflow | runner | passed | median s | failed checks | time.clock_s | time.idle_s |
|---|---|---|---|---|---|---|
| image_and_read_back | general | 1/1 | 92 | - | 0.518 | 35.7 |
| image_at_field_of_view | general | 1/1 | 54 | - | 0.518 | 0.00779 |
| set_defocus | general | 1/1 | 35 | - | - | 0.00608 |
| move_and_image | general | 1/1 | 47 | - | 1.59 | 0.0422 |
| read_status_only | general | 1/1 | 27 | - | 0 | 0.0191 |
| blank_the_beam | general | 1/1 | 8 | - | - | 0.0062 |
| eds_spectrum_elements | general | 1/1 | 61 | - | 0.2 | 8.25 |
| compare_two_regions | general | 1/1 | 178 | - | 5.25 | 25.7 |
| describe_latest_image | general | 1/1 | 86 | - | - | - |
| count_tiles_in_run | general | 1/1 | 54 | - | - | - |
| short_exposure_spectrum | general | 1/1 | 58 | - | - | - |
| points_of_interest_on_map | general | 1/1 | 207 | - | - | - |
| find_particle_then_eds | general | 1/1 | 77 | - | 0.718 | 1.2 |
| survey_rank_and_analyse | general | 1/1 | 208 | - | 43.3 | 53.7 |
| focus_series | general | 1/1 | 176 | - | 2.07 | 109 |
| map_300nm | general | 1/1 | 84 | - | 6.15 | 6.07 |
| map_600nm_with_eds | general | 1/1 | 156 | - | 40.7 | 6.87 |
| cuj2_map_with_eds_points | general | 1/1 | 86 | - | 22.9 | 6.7 |
| cuj3_three_regions_high_mag | general | 0/1 | 182 | scored_regions | 3.22 | 13.2 |
| map_1um_overview | general | 1/1 | 124 | - | 55.9 | 0.27 |
| map_and_stitch | general | 1/1 | 77 | - | 24.7 | 0.128 |
| explore_then_zoom | general | 1/1 | 149 | - | 24.7 | 8.2 |
| abort_then_resume | general | 1/1 | 817 | - | 503 | 29.7 |
| revisit_point_of_interest | general | 1/1 | 89 | - | 0 | 0.0619 |
| zoom_to_atomic_columns | general | 1/1 | 142 | - | 1.04 | 0.0299 |
| held_move_is_approved | general | 1/1 | 78 | - | 1.03 | 38.4 |
| held_move_is_denied | general | 1/1 | 15 | - | - | 0.133 |
| held_map_is_approved | general | 1/1 | 117 | - | 6.15 | 38.1 |
| field_of_view_out_of_range | general | 1/1 | 44 | - | - | 0.00746 |
| dose_budget_refused | general | 1/1 | 126 | - | 0 | 102 |
| stage_out_of_range | general | 1/1 | 12 | - | - | 0.00625 |
| recover_from_unknown_detector | general | 1/1 | 74 | - | 1.04 | 41.8 |

Passed 31 of 32 runs; sleepwalking calls: 0.
