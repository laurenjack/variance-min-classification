# Git Add & Commit

This command helps stage and commit changes to a git repository (without pushing).

## Process

1. **Determine the target repository and branch**
   - Check the current working repository and branch using `git branch --show-current`
   - Inform the user: "I'll commit to `<repo>` on branch `<branch_name>`."

2. **Show pending changes**
   - Run `git status` to show what will be committed
   - If no changes, inform the user and stop

3. **Determine commit message**
   - Analyze the changes (run `git diff --staged` after staging if needed)
   - Write a logical, descriptive commit message based on what changed
   - Do NOT ask the user for a commit message - decide on one yourself

4. **Stage changes**
   ```bash
   git add -A
   ```

5. **Commit changes**
   ```bash
   git commit -m "<commit_message>"
   ```

   ⚠️ **Pre-commit hooks**: The repository may have pre-commit hooks that run (linting, formatting, etc.). This is normal - let them complete.

6. **Confirm success**
   - Show the commit hash and summary

## Commit Message Guidelines

- Start with a brief summary line (imperative mood, e.g., "Add", "Fix", "Update")
- If the change is substantial, add a blank line and bullet points explaining key changes
- Be specific about what changed, not vague ("Update files" is bad, "Add user authentication endpoint" is good)

## Example Interaction

> **User**: /commit
> **Agent**: I'll commit to `catalog-relevance` on branch `top-k-fdna-mean`.
>
> Current changes:
> - modified: catalog_relevance/datasets/registry.py
> - modified: scripts/manual_job_runs/run_job.sh
>
> *stages and commits with message: "Update dataset version and fix run_job script path"*
> ✅ Committed successfully! (abc1234)

## Notes

- Always use `git add -A` to stage all changes (new, modified, deleted files)
- This command does NOT push - use `/push` if you also want to push to remote
