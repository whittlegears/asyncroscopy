# Workflow benchmark: qwen3:8b, stack twin, 0253e3c

| workflow | runner | passed | median s | failed checks | time.clock_s | time.idle_s |
|---|---|---|---|---|---|---|
| eds_spectrum_elements | general | 2/3 | 70 | answer_eds_consistent | 1 | 9.95 |
| short_exposure_spectrum | general | 3/3 | 26 | - | - | - |
| find_particle_then_eds | general | 3/3 | 43 | - | 0.718 | 1.88 |
| survey_rank_and_analyse | general | 3/3 | 225 | - | 43.3 | 37.1 |
| map_600nm_with_eds | general | 3/3 | 142 | - | 40.7 | 8.16 |
| cuj2_map_with_eds_points | general | 3/3 | 88 | - | 22.9 | 8.05 |
| spectrum_without_touching | general | 3/3 | 35 | - | 1 | 5.57 |

Passed 20 of 21 runs; sleepwalking calls: 0.
