# Review Branch

Reviews a branch by comparing it against main/master.

**Usage**: `/review <branch_name> [base_branch]`

- `branch_name` (required): The branch to review
- `base_branch` (optional): The branch to compare against. If not provided, auto-detect main or master.

## Process

1. **Ask for PR documentation**
   > "Before I review, can you provide context about this PR? You can:
   > - Paste the PR description
   > - Point me to a file that describes the changes
   > - Tell me briefly what the PR is trying to achieve"
   
   If user says "just review it" without context, ask:
   > "What is this PR trying to accomplish? A brief description will help me review more effectively."

2. **Fetch latest and compare**
   ```bash
   git fetch origin <branch_name>
   git fetch origin <base_branch>
   git diff origin/<base_branch>...origin/<branch_name> --stat
   ```

3. **Assess PR size**
   - If the diff is large (many files or significant line changes), **write a review plan first**:
     > "This is a large PR. Here's my review plan:
     > 1. Core logic changes in X
     > 2. Test changes in Y
     > 3. Config/infrastructure changes
     > 
     > I'll review in this order. Let me know if you want me to prioritize differently."

4. **Review the changes**
   - Read the diff and changed files
   - Compare against the provided documentation/intent

5. **Provide summary**

## Review Focus

✅ **Do check for:**
- Alignment with stated PR intent/documentation
- Bugs and potential errors
- Risks (edge cases, breaking changes, security concerns)

❌ **Don't worry about:**
- Style issues (unless really unusual)
- Minor formatting

## Output Format

Keep the summary **concise and to the point**:

```
## Review Summary: <branch_name>

**Alignment**: ✅ Matches intent / ⚠️ Partial / ❌ Deviates from docs

**Issues Found**:
- 🔴 [Critical] Description of critical issue
- 🟡 [Warning] Description of potential problem  
- 🟢 [Note] Minor observation

**Risks**:
- Any edge cases or breaking changes identified

**Verdict**: Ready to merge / Needs changes / Needs discussion
```

## Example Interaction

> **User**: /review feature-new-auth  
> **Agent**: Before I review, can you provide context about this PR?  
> **User**: It adds OAuth2 support to the login flow  
> **Agent**: *fetches branches, reads diff*  
>
> ## Review Summary: feature-new-auth
> **Alignment**: ✅ Matches intent - OAuth2 flow implemented  
> **Issues Found**:
> - 🔴 [Critical] Token refresh not handled in `auth.py:45`
> - 🟡 [Warning] Error message exposes internal details in `login.py:78`
>
> **Risks**: Breaking change for existing session handling  
> **Verdict**: Needs changes

## Notes

- Always get context before reviewing - blind reviews miss intent
- Report issues but let the user decide on fixes
- For very large PRs, plan first to manage context effectively

