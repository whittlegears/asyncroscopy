# Workflow benchmark: qwen3:8b, stack twin, 8539eb6

| workflow | runner | passed | median s | failed checks |
|---|---|---|---|---|
| image_and_read_back | general | 1/1 | 55 | - |
| image_at_field_of_view | general | 1/1 | 38 | - |
| set_defocus | general | 1/1 | 5 | - |
| move_and_image | general | 1/1 | 18 | - |
| read_status_only | general | 1/1 | 20 | - |
| blank_the_beam | general | 1/1 | 4 | - |
| eds_spectrum_elements | general | 1/1 | 74 | - |
| compare_two_regions | general | 1/1 | 133 | - |
| describe_latest_image | general | 1/1 | 88 | - |
| count_tiles_in_run | general | 1/1 | 17 | - |
| short_exposure_spectrum | general | 1/1 | 20 | - |
| points_of_interest_on_map | general | 1/1 | 407 | - |
| find_particle_then_eds | general | 1/1 | 107 | - |
| survey_rank_and_analyse | general | 0/1 | 225 | answer_eds_consistent, called |
| focus_series | general | 0/1 | 74 | param_near |
| map_300nm | general | 1/1 | 100 | - |
| map_600nm_with_eds | general | 1/1 | 182 | - |
| cuj2_map_with_eds_points | general | 1/1 | 83 | - |
| cuj3_three_regions_high_mag | general | 0/1 | 129 | images_linked_to_regions, scored_regions, visited_regions |
| map_1um_overview | general | 1/1 | 180 | - |
| map_and_stitch | general | 1/1 | 89 | - |
| explore_then_zoom | general | 1/1 | 180 | - |
| abort_then_resume | general | 1/1 | 1344 | - |
| revisit_point_of_interest | general | 1/1 | 80 | - |
| held_move_is_approved | general | 1/1 | 69 | - |
| held_move_is_denied | general | 1/1 | 14 | - |
| held_map_is_approved | general | 1/1 | 148 | - |
| field_of_view_out_of_range | general | 1/1 | 38 | - |
| dose_budget_refused | general | 1/1 | 28 | - |
| stage_out_of_range | general | 1/1 | 12 | - |
| recover_from_unknown_detector | general | 1/1 | 45 | - |

Passed 28 of 31 runs; sleepwalking calls: 5.
