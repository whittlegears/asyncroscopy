# Workflow benchmark: qwen3:8b, stack twin, 54d25dd

| workflow | runner | passed | median s | failed checks |
|---|---|---|---|---|
| image_and_read_back | general | 1/1 | 60 | - |
| image_at_field_of_view | general | 1/1 | 41 | - |
| set_defocus | general | 1/1 | 6 | - |
| move_and_image | general | 1/1 | 23 | - |
| read_status_only | general | 1/1 | 23 | - |
| blank_the_beam | general | 1/1 | 4 | - |
| eds_spectrum_elements | general | 1/1 | 77 | - |
| compare_two_regions | general | 1/1 | 142 | - |
| describe_latest_image | general | 1/1 | 114 | - |
| count_tiles_in_run | general | 1/1 | 18 | - |
| short_exposure_spectrum | general | 1/1 | 22 | - |
| points_of_interest_on_map | general | 1/1 | 408 | - |
| find_particle_then_eds | general | 1/1 | 125 | - |
| survey_rank_and_analyse | general | 0/1 | 248 | answer_eds_consistent |
| focus_series | general | 0/1 | 108 | param_near |
| map_300nm | general | 1/1 | 104 | - |
| map_600nm_with_eds | general | 1/1 | 178 | - |
| cuj2_map_with_eds_points | general | 1/1 | 96 | - |
| cuj3_three_regions_high_mag | general | 0/1 | 141 | images_linked_to_regions, scored_regions, visited_regions |
| map_1um_overview | general | 1/1 | 188 | - |
| map_and_stitch | general | 1/1 | 89 | - |
| explore_then_zoom | general | 0/1 | 180 | no_sleepwalking, zoomed_on_region |
| abort_then_resume | general | 1/1 | 1343 | - |
| revisit_point_of_interest | general | 1/1 | 83 | - |
| held_move_is_approved | general | 1/1 | 70 | - |
| held_move_is_denied | general | 1/1 | 16 | - |
| held_map_is_approved | general | 1/1 | 158 | - |
| field_of_view_out_of_range | general | 1/1 | 40 | - |
| dose_budget_refused | general | 1/1 | 66 | - |
| stage_out_of_range | general | 1/1 | 13 | - |
| recover_from_unknown_detector | general | 1/1 | 45 | - |

Passed 27 of 31 runs; sleepwalking calls: 6.
