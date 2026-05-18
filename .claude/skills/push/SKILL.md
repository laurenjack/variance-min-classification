---
name: push
description: Stage, commit, and push changes to the current git repository's remote.
---

# Push

Stage, commit, and push.

## Process

1. **Confirm target**: run `git branch --show-current` and ask "I'll push to `<repo>` on branch `<branch>`. Is that correct?"
   - If yes, proceed.
   - If no, ask which repo/branch.
2. **Commit**: follow the `commit` skill — show `git status`, stage with `git add -A`, write a message yourself (do not ask), commit. Let pre-commit hooks complete.
3. **Push**: `git push origin <branch>`.

## Conflicts

If the push is rejected because remote has commits you don't have, **hand back to the user**. Do not auto-resolve:

> The push was rejected — remote has commits you don't. I recommend you take over:
> - `git pull origin <branch>` to merge
> - `git pull --rebase origin <branch>` to rebase
> - `git push --force origin <branch>` to overwrite (use with caution)
>
> Let me know once it's resolved.

## Notes

- Never force-push without explicit user confirmation.
- Use the `commit` skill if you only want to commit.
