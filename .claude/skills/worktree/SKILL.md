---
name: worktree
description: Create a git worktree from the current branch, make changes, run tests, then merge back atomically.
---

# Worktree

Create a worktree from the current branch and make the requested changes there.

**Never merge back to the parent branch until the entire task is complete and any associated tests pass.** This includes intermediate "tests-first" steps: even if a failing test suite is committed deliberately (so the agent can verify its fix later), keep that commit on the worktree branch only — never merge it back. The parent must always reflect a green, complete state.

## Sequence

1. Create the worktree on a new branch from the parent.
2. Make changes, run iterations, commit on the worktree branch.
3. Verify all relevant tests pass — the suite the user cares about, not just unit tests.
4. Run the `commit` skill for the final commit if needed.
5. Only then proceed to the merge step below.

If the user interrupts before tests are green, leave the worktree alive and report status. Do not merge a half-finished or red-suite branch.

## Merging back

Use an atomic file-lock to coordinate with other agents:

```bash
mkdir locks/integration.lock     # atomic — fails if another agent has the lock
# ... if it fails, sleep briefly and retry ...
git merge <worktree-branch>      # resolve conflicts using commit history
rmdir locks/integration.lock     # always release
```

## Cleanup after merge

1. Remove the worktree: `git worktree remove <path>` (use `--force` if needed)
2. Prune stale references: `git worktree prune`
3. Delete the branch: `git branch -D <branch>` (safe — already merged)
