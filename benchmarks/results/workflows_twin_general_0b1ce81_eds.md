# Workflow benchmark: qwen3:8b, stack twin, ff74972

| workflow | runner | passed | median s | failed checks | time.clock_s | time.idle_s |
|---|---|---|---|---|---|---|
| eds_spectrum_elements | general | 3/3 | 35 | - | 1 | 9.48 |
| short_exposure_spectrum | general | 3/3 | 22 | - | - | - |
| find_particle_then_eds | general | 3/3 | 41 | - | 0.718 | 1.2 |
| survey_rank_and_analyse | general | 3/3 | 198 | - | 43.3 | 33.2 |
| map_600nm_with_eds | general | 3/3 | 108 | - | 40.7 | 6.72 |
| cuj2_map_with_eds_points | general | 3/3 | 84 | - | 22.9 | 6.65 |
| spectrum_without_touching | general | 3/3 | 39 | - | 1 | 0.00678 |

Passed 21 of 21 runs; sleepwalking calls: 0.
