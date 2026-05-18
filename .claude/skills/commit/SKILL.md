---
name: commit
description: Stage and commit changes to the current git repository without pushing. Writes the commit message automatically.
---

# Commit

Stage and commit changes. Do not push — use the `push` skill for that.

## Process

1. **Report target**: run `git branch --show-current` and tell the user "I'll commit to `<repo>` on branch `<branch>`."
2. **Show pending changes**: `git status`. If no changes, say so and stop.
3. **Decide a commit message yourself** (do NOT ask the user):
   - Run `git diff` / `git diff --staged` to understand the change.
   - Imperative mood, brief summary line. If substantial, add a blank line and bullet points.
   - Be specific: "Add user authentication endpoint" — not "Update files".
4. **Stage all changes**: `git add -A`.
5. **Commit**: `git commit -m "<message>"`. Pre-commit hooks may run — let them complete.
6. **Confirm** with commit hash and summary.

## Notes

- `git add -A` covers new, modified, and deleted files.
- This skill does NOT push. Use `push` if you want to push too.
