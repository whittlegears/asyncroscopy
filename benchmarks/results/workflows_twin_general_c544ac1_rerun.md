# Workflow benchmark: qwen3:8b, stack twin, c544ac1

| workflow | runner | passed | median s | failed checks |
|---|---|---|---|---|
| points_of_interest_on_map | general | 0/1 | 9 | answer_number_near |
| survey_rank_and_analyse | general | 0/1 | 258 | called |
| focus_series | general | 0/1 | 47 | param_near |
| cuj3_three_regions_high_mag | general | 1/1 | 108 | - |
| abort_then_resume | general | 0/1 | 134 | answered, resumed_after_abort |
| revisit_point_of_interest | general | 1/1 | 29 | - |

Passed 2 of 6 runs; sleepwalking calls: 0.
