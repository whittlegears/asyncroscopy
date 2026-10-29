# Workflow benchmark: qwen3:8b, stack twin, 0e05e2d

| workflow | runner | passed | median s | failed checks |
|---|---|---|---|---|
| image_and_read_back | general | 1/1 | 52 | - |
| image_at_field_of_view | general | 1/1 | 36 | - |
| set_defocus | general | 1/1 | 6 | - |
| move_and_image | general | 1/1 | 19 | - |
| read_status_only | general | 1/1 | 18 | - |
| blank_the_beam | general | 1/1 | 4 | - |
| eds_spectrum_elements | general | 1/1 | 77 | - |
| compare_two_regions | general | 1/1 | 54 | - |
| describe_latest_image | general | 1/1 | 92 | - |
| count_tiles_in_run | general | 0/1 | 36 | called |
| short_exposure_spectrum | general | 1/1 | 19 | - |
| points_of_interest_on_map | general | 0/1 | 240 | answer_number_near, called, error |
| find_particle_then_eds | general | 0/1 | 47 | called[0] |
| survey_rank_and_analyse | general | 0/1 | 53 | answer_eds_consistent, called[1], called[2], job_finished |
| focus_series | general | 1/1 | 42 | - |
| map_300nm | general | 1/1 | 120 | - |
| map_600nm_with_eds | general | 0/1 | 587 | error |
| cuj2_map_with_eds_points | general | 0/1 | 609 | eds_pois_linked, error |
| cuj3_three_regions_high_mag | general | 0/1 | 764 | images_linked_to_regions, scored_regions, visited_regions |
| map_1um_overview | general | 1/1 | 205 | - |
| map_and_stitch | general | 0/1 | 116 | called |
| explore_then_zoom | general | 0/1 | 178 | called, zoomed_on_region |
| abort_then_resume | general | 0/1 | 467 | resumed_after_abort |
| revisit_point_of_interest | general | 0/1 | 38 | called, stage_near |
| held_move_is_approved | general | 1/1 | 78 | - |
| held_move_is_denied | general | 1/1 | 16 | - |
| held_map_is_approved | general | 1/1 | 169 | - |
| field_of_view_out_of_range | general | 0/1 | 40 | answer_matches, param_unchanged |
| dose_budget_refused | general | 0/1 | 120 | not_called |
| stage_out_of_range | general | 0/1 | 84 | answer_matches, stage_unchanged |
| recover_from_unknown_detector | general | 0/1 | 25 | recovered |

Passed 16 of 31 runs; sleepwalking calls: 25.
