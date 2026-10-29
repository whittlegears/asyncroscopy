"""Pure, backend-agnostic routines: no Tango, no instrument, only numbers in and numbers out.

Jobs (``asyncroscopy.jobs``) call these to decide what to do next; tests and
benches call them directly. See docs/designs/concepts_brief.md, decision D1.
"""
