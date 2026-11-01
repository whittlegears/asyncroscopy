# Workflow benchmark: qwen3:8b, stack twin, 93e43c1

| workflow | runner | passed | median s | failed checks | time.clock_s | time.idle_s |
|---|---|---|---|---|---|---|
| image_and_read_back | general | 0/2 | 73 | called, viewed_own_data | 0 | 32.2 |
| image_at_field_of_view | general | 2/2 | 28 | - | 0.0361 | 0.00758 |
| eds_spectrum_elements | general | 2/2 | 37 | - | 0.2 | 9.66 |
| compare_two_regions | general | 2/2 | 141 | - | - | 0.0053 |
| describe_latest_image | general | 2/2 | 51 | - | - | - |
| short_exposure_spectrum | general | 2/2 | 48 | - | - | - |
| find_particle_then_eds | general | 2/2 | 43 | - | 0.718 | 1.26 |
| focus_series | general | 1/2 | 143 | called[1] | 1.55 | 103 |

Passed 13 of 16 runs; sleepwalking calls: 0.
