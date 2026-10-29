---
name: recover_from_errors
description: Read a refused or failed call, fix its arguments and try once more; never repeat an identical failing call.
version: 1.0.0
min_asyncroscopy: 0.1.0
keywords: [error, failed, refused, retry, wrong, fix, broken, not working]
acts: false
tools: []
---

# When a call fails

1. Read the whole message. It usually names the problem and the fix: a unit (lengths are metres), a missing field, an out-of-range value with its limit, a key that does not exist (it lists close keys), or the right tool to use instead.
2. Change what the message says and call again. Never send the same call with the same arguments after it failed: the server counts repeats and you will get the same answer.
3. A `plan_rejected` with `violations` lists each step outside a limit: shrink or move the area, then reserve again.
4. "busy": another job holds the device. Wait for it (`wait_for_job`) or tell the user; do not release someone else's reservation.
5. "Tiled ... not answering" or "try again in a few seconds": wait once and retry once. If it fails again, report it.
6. A job that ends `failed` reports `last_error`: report it; start it again only if the error says the cause is gone.
7. After two different fixes fail, stop and tell the user exactly what was refused and why.
