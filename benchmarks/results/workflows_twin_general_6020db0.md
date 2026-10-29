# Workflow benchmark: qwen3:8b, stack twin, 6020db0

| workflow | runner | passed | median s | failed checks |
|---|---|---|---|---|
| image_and_read_back | general | 1/1 | 54 | - |
| image_at_field_of_view | general | 1/1 | 38 | - |
| set_defocus | general | 1/1 | 6 | - |
| move_and_image | general | 1/1 | 20 | - |
| read_status_only | general | 1/1 | 18 | - |
| blank_the_beam | general | 1/1 | 4 | - |
| eds_spectrum_elements | general | 1/1 | 73 | - |
| compare_two_regions | general | 0/1 | 480 | answer_matches, answered, error |
| describe_latest_image | general | 1/1 | 93 | - |
| count_tiles_in_run | general | 0/0 | 0 | error |
| short_exposure_spectrum | general | 0/0 | 0 | error |
| points_of_interest_on_map | general | 0/0 | 0 | error |
| find_particle_then_eds | general | 1/1 | 104 | - |
| survey_rank_and_analyse | general | 0/1 | 240 | answer_eds_consistent, called[1] |
| focus_series | general | 0/1 | 42 | param_near |
| map_300nm | general | 1/1 | 484 | - |
| map_600nm_with_eds | general | 1/1 | 220 | - |
| cuj2_map_with_eds_points | general | 1/1 | 82 | - |
| cuj3_three_regions_high_mag | general | 0/1 | 301 | images_linked_to_regions, scored_regions, visited_regions |
| map_1um_overview | general | 1/1 | 225 | - |
| map_and_stitch | general | 1/1 | 134 | - |
| explore_then_zoom | general | 1/1 | 205 | - |
| abort_then_resume | general | 0/1 | 440 | resumed_after_abort |
| revisit_point_of_interest | general | 0/0 | 0 | error |
| held_move_is_approved | general | 1/1 | 66 | - |
| held_move_is_denied | general | 1/1 | 15 | - |
| held_map_is_approved | general | 1/1 | 132 | - |
| field_of_view_out_of_range | general | 0/1 | 40 | param_unchanged |
| dose_budget_refused | general | 1/1 | 33 | - |
| stage_out_of_range | general | 1/1 | 32 | - |
| recover_from_unknown_detector | general | 1/1 | 45 | - |

Passed 21 of 27 runs; sleepwalking calls: 6.
