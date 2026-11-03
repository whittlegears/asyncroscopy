# Workflow benchmark: qwen3:8b, stack twin, c897032

| workflow | runner | passed | median s | failed checks | time.clock_s | time.idle_s |
|---|---|---|---|---|---|---|
| eds_spectrum_elements | general | 3/3 | 49 | - | 1 | 9.48 |
| short_exposure_spectrum | general | 3/3 | 21 | - | - | - |
| find_particle_then_eds | general | 3/3 | 43 | - | 0.718 | 1.21 |
| survey_rank_and_analyse | general | 3/3 | 211 | - | 43.3 | 34.5 |
| spectrum_without_touching | general | 0/3 | 79 | answer_eds_consistent | 1 | 0.00605 |

Passed 12 of 15 runs; sleepwalking calls: 0.
