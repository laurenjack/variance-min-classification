# Git Add, Commit & Push

This command helps stage, commit, and push changes to a git repository.

## Process

1. **Determine the target repository and branch**
   - Check the current working repository and branch using `git branch --show-current`
   - **Always confirm with the user**: "I'll push to `<repo>` on branch `<branch_name>`. Is that correct?"
   - If **yes**: proceed
   - If **no**: check if they provided the correct repo/branch in their response. If not, ask: "Which repository and branch should I push to?"

2. **Commit changes**
   - Follow the `/commit` process (see commit.md):
     - Show pending changes with `git status`
     - If no changes, inform the user and stop
     - Stage all changes with `git add -A`
     - Determine a logical commit message automatically (do NOT ask the user)
     - Commit the changes
   - ⚠️ **Pre-commit hooks**: Let them complete and continue to the push step regardless of hook output.

3. **Push changes**
   ```bash
   git push origin <branch_name>
   ```

## Handling Conflicts

If the push fails due to conflicts (remote has changes not in local):

**Always suggest the user take over and inspect themselves.**

> "The push was rejected because the remote branch has commits that your local branch doesn't have. I recommend you take over from here to inspect the changes and resolve this manually. You can:
> - `git pull origin <branch_name>` to merge remote changes
> - `git pull --rebase origin <branch_name>` to rebase your changes on top
> - `git push --force origin <branch_name>` to overwrite remote (use with caution)
>
> Let me know once you've resolved it, or if you'd like help with something else."

Do not attempt to resolve conflicts automatically - this requires human judgment to avoid losing work.

## Example Interaction

> **User**: /push
> **Agent**: I'll push to `catalog-relevance` on branch `top-k-fdna-mean`. Is that correct?
> **User**: Yes
> **Agent**: Current changes:
> - modified: catalog_relevance/datasets/registry.py
> - new file: scripts/run_eval.py
>
> *stages, commits with message "Add evaluation script and update registry", then pushes*
> ✅ Changes pushed successfully!

> **User**: Push my changes
> **Agent**: I'll push to `catalog-relevance` on branch `main`. Is that correct?
> **User**: No, push to lounge-atlas on feature-branch
> **Agent**: Got it, pushing to `lounge-atlas` on branch `feature-branch`...

## Notes

- Always use `git add -A` to stage all changes (new, modified, deleted files)
- Never force push without explicit user confirmation
- Use `/commit` if you only want to commit without pushing
