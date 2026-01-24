# Checkout New Branch

Creates and switches to a new branch.

**Usage**: `/checkout <branch_name>`

## Process

1. **Check current state**
   ```bash
   git status
   git branch --show-current
   ```

2. **Evaluate conditions**

   | On master/main? | Uncommitted changes? | Action |
   |-----------------|---------------------|--------|
   | ✓ Yes | ✗ No | ✅ Proceed automatically |
   | ✗ No | - | ⚠️ Ask user |
   | - | ✓ Yes | ⚠️ Ask user |

3. **If conditions allow automatic checkout**
   ```bash
   git checkout -b <branch_name>
   ```

4. **If not on master/main** - Ask:
   > "You're currently on branch `<current_branch>`. Do you want to:
   > - Create `<branch_name>` from `<current_branch>`?
   > - Switch to main/master first, then create `<branch_name>`?"

5. **If uncommitted changes exist** - Ask:
   > "You have uncommitted changes. Do you want to:
   > - Stash them before creating the new branch?
   > - Commit them first? (I can help with /push)
   > - Carry them over to the new branch?"

## Example Interactions

> **User**: /checkout feature-new-model  
> **Agent**: *checks status - on main, no changes*  
> ✅ Created and switched to branch `feature-new-model`

> **User**: /checkout feature-new-model  
> **Agent**: You're currently on branch `bugfix-123`. Do you want to create `feature-new-model` from here, or switch to main first?

> **User**: /checkout feature-new-model  
> **Agent**: You have uncommitted changes (3 files modified). Do you want to stash them, commit them first, or carry them to the new branch?

## Notes

- Always check both conditions (branch + changes) before proceeding
- The branch name is required - if not provided, ask for it

