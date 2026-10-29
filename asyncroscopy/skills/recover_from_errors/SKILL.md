---
name: recover_from_errors
description: Read a refused or failed call, fix its arguments and try once more; never repeat an identical failing call, never change a value the user stated.
version: 1.1.0
min_asyncroscopy: 0.1.0
keywords: [error, failed, refused, retry, wrong, fix, broken, not working, timeout, timed out]
acts: false
tools: []
---

# When a call fails

1. Read the whole message. It usually names the problem and the fix: a unit (lengths are metres), a missing field, an out-of-range value with its limit, a key that does not exist (it lists close keys), or the right tool to use instead.
2. Fix the form, never the user's numbers. A field name, a unit conversion or a JSON shape may change. A value the user stated (a field of view, a size, a position, a dose limit, an exposure) may not: if the instrument refuses it, tell the user the limit the message gives and ask what to do. Never pick another value yourself and report success.
3. Never send the same call with the same arguments after it failed: the server counts repeats and you will get the same answer.
4. A call that timed out while moving or acquiring may still be running. Before anything else, read the state (`get_stage`, `get_parameters`, `wait_for_job`) and never resend the same motion or acquisition until the state shows it ended.
5. Never "reset" by unblanking the beam, toggling settings or moving the stage: those act on the sample. Only what the user asked for, or a stop (abort, blank), is allowed.
6. A `plan_rejected` with `violations` lists each step outside a limit: tell the user the limit; change the plan only in ways the request allows (for example the position when the user gave none).
7. "busy": another job holds the device. Wait for it (`wait_for_job`) or tell the user; do not release someone else's reservation.
8. "Tiled ... not answering" or "try again in a few seconds": wait once and retry once. If it fails again, report it.
9. A job that ends `failed` reports `last_error`: report it; start it again only if the error says the cause is gone.
10. After two different fixes fail, stop and tell the user exactly what was refused and why.
