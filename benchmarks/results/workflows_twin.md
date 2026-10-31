# Workflow benchmark: qwen3:8b, stack twin, c4d2f0b

| workflow | runner | passed | median s | failed checks | time.clock_s | time.idle_s |
|---|---|---|---|---|---|---|
| points_of_interest_on_map | general | 0/1 | 56 | answer_number_near | - | - |
| cuj2_map_with_eds_points | general | 1/1 | 184 | - | 22.9 | 9.67 |
| cuj3_three_regions_high_mag | general | 1/1 | 255 | - | 3.21 | 14.2 |
| map_and_stitch | general | 1/1 | 174 | - | 24.7 | 8.04 |
| explore_then_zoom | general | 0/1 | 230 | answered | 24.7 | 11.5 |
| revisit_point_of_interest | general | 0/1 | 22 | answered, called, stage_near | - | - |

Passed 3 of 6 runs; sleepwalking calls: 0.
