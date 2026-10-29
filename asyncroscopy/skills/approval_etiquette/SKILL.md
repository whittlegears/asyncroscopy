---
name: approval_etiquette
description: What to do when a call is held for an operator (APPROVAL REQUIRED), approved, denied or expired.
version: 1.0.0
min_asyncroscopy: 0.1.0
keywords: [approval, approve, held, operator, denied, permission]
requires: [approval_status]
acts: false
tools: [approval_status]
---

# Held calls

In approve mode the instrument holds high-risk calls (moves, acquisitions, starting jobs) for a person.

1. A reply starting `APPROVAL REQUIRED` carries a `request_id`. Nothing ran.
2. Call `approval_status` with that `request_id` every few seconds. Do not call anything else that moves or acquires meanwhile.
3. `approved`: repeat exactly the same call, once. The approval is single-use.
4. `denied` or `expired`: do not retry, do not try another way to do the same thing. Release any reservation you hold and tell the user what was denied and the reason given.
5. Never try to change the approval mode or approve your own request: those tools are the operator's.

Reading (get_*, list_*, view_data, job status) is never held.
