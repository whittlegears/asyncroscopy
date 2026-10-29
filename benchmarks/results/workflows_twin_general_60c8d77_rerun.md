# Workflow benchmark: qwen3:8b, stack twin, 60c8d77

| workflow | runner | passed | median s | failed checks |
|---|---|---|---|---|
| compare_two_regions | general | 1/1 | 168 | - |
| count_tiles_in_run | general | 1/1 | 71 | - |
| short_exposure_spectrum | general | 1/1 | 23 | - |
| points_of_interest_on_map | general | 0/1 | 8 | answer_number_near |
| survey_rank_and_analyse | general | 0/1 | 243 | answer_eds_consistent |
| focus_series | general | 0/1 | 52 | param_near |
| cuj3_three_regions_high_mag | general | 0/1 | 162 | images_linked_to_regions, visited_regions |
| abort_then_resume | general | 0/1 | 255 | answered, resumed_after_abort |
| revisit_point_of_interest | general | 0/1 | 82 | no_sleepwalking |
| field_of_view_out_of_range | general | 1/1 | 39 | - |

Passed 4 of 10 runs; sleepwalking calls: 1.
