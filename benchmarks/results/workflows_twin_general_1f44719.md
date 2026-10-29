# Workflow benchmark: qwen3:8b, stack twin, 1f44719

| workflow | runner | passed | median s | failed checks |
|---|---|---|---|---|
| image_and_read_back | general | 1/1 | 59 | - |
| image_at_field_of_view | general | 1/1 | 40 | - |
| set_defocus | general | 1/1 | 35 | - |
| move_and_image | general | 1/1 | 38 | - |
| read_status_only | general | 1/1 | 42 | - |
| blank_the_beam | general | 1/1 | 9 | - |
| eds_spectrum_elements | general | 1/1 | 88 | - |
| compare_two_regions | general | 1/1 | 145 | - |
| describe_latest_image | general | 1/1 | 118 | - |
| count_tiles_in_run | general | 1/1 | 22 | - |
| short_exposure_spectrum | general | 1/1 | 49 | - |
| points_of_interest_on_map | general | 0/1 | 81 | answer_number_near |
| find_particle_then_eds | general | 0/1 | 80 | answer_eds_consistent, called[1] |
| survey_rank_and_analyse | general | 0/1 | 287 | answer_eds_consistent |
| focus_series | general | 0/1 | 118 | param_near |
| map_300nm | general | 1/1 | 100 | - |
| map_600nm_with_eds | general | 1/1 | 217 | - |
| cuj2_map_with_eds_points | general | 1/1 | 99 | - |
| cuj3_three_regions_high_mag | general | 1/1 | 204 | - |
| map_1um_overview | general | 1/1 | 207 | - |
| map_and_stitch | general | 1/1 | 104 | - |
| explore_then_zoom | general | 1/1 | 183 | - |
| abort_then_resume | general | 1/1 | 1381 | - |
| revisit_point_of_interest | general | 1/1 | 82 | - |
| held_move_is_approved | general | 1/1 | 70 | - |
| held_move_is_denied | general | 1/1 | 16 | - |
| held_map_is_approved | general | 1/1 | 154 | - |
| field_of_view_out_of_range | general | 1/1 | 41 | - |
| dose_budget_refused | general | 1/1 | 130 | - |
| stage_out_of_range | general | 1/1 | 14 | - |
| recover_from_unknown_detector | general | 1/1 | 47 | - |

Passed 27 of 31 runs; sleepwalking calls: 5.
