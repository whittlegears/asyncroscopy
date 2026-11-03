# Workflow benchmark: qwen3:8b, stack twin, c897032

| workflow | runner | passed | median s | failed checks | time.clock_s | time.idle_s |
|---|---|---|---|---|---|---|
| field_of_view_in_micrometres | general | 3/3 | 6 | - | - | 0.00605 |
| defocus_in_angstrom | general | 3/3 | 6 | - | - | 0.00654 |
| move_relative_to_here | general | 3/3 | 10 | - | - | 0.0109 |
| image_then_put_fov_back | general | 3/3 | 29 | - | 0.518 | 15 |
| spectrum_without_touching | general | 3/3 | 35 | - | - | 0.00594 |
| follow_up_doubles_the_field | general | 3/3 | 64 | - | 1.04 | 37.4 |
| question_needs_no_action | general | 3/3 | 12 | - | - | 0.00681 |

Passed 21 of 21 runs; sleepwalking calls: 3.
