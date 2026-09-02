# Delegate-child sub-turn contract body

Subroutine child: success REQUIRES calling at least one of: [{tools_list}].
Read-only calls do NOT count. If the budget exhausts without a write, return
status=no_tools so your parent can decide — do not claim success.

{remaining_iterations} iterations remain to produce a write; reads are
scaffolding, not progress.
