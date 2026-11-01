# Workflow benchmark: qwen3:8b, stack twin, 93e43c1

| workflow | runner | passed | median s | failed checks | time.clock_s | time.idle_s |
|---|---|---|---|---|---|---|
| image_and_read_back | general | 0/2 | 60 | viewed_own_data | - | 0.00781 |
| image_at_field_of_view | general | 2/2 | 57 | - | 0.518 | 0.0183 |
| eds_spectrum_elements | general | 1/2 | 59 | answer_eds_consistent | 1 | 9.09 |
| compare_two_regions | general | 2/2 | 167 | - | - | 0.0174 |
| describe_latest_image | general | 2/2 | 103 | - | - | - |
| short_exposure_spectrum | general | 2/2 | 68 | - | - | - |
| find_particle_then_eds | general | 2/2 | 79 | - | 0.718 | 1.68 |
| focus_series | general | 2/2 | 199 | - | 1.55 | 120 |

Passed 13 of 16 runs; sleepwalking calls: 0.
