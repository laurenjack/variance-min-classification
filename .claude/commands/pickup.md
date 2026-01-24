# Pickup Task

This command picks up a specific task from the TODO list and works through it.

## Usage

```
/pickup <task_number>
```

**The task number is mandatory.** This command will not work without specifying which task to pick up.

## Arguments

The argument passed to this command is: $ARGUMENTS

**If no argument is provided, stop immediately and ask the user to specify a task number.** Do not proceed with any task lookup or execution.

## Process

1. **Validate the argument**
   - If `$ARGUMENTS` is empty or not provided, respond with: "Please specify a task number. Usage: `/pickup <task_number>`" and stop.
   - Parse the task number from the argument

2. **Read the TODO file**
   - Open `orc/TODO.md`
   - Find the task with the specified number

3. **Validate the task**
   - If the specified task number doesn't exist, inform the user and stop
   - If the task already has a ✓ (tick) or ✗ (cross) at the end, inform the user it's already completed and stop

4. **Understand the task**
   - Read the task description carefully
   - Explore the documentation and code that is relevant
   - Identify any ambiguities or missing information

5. **Ask clarification questions**
   - Before starting work, ask the user any questions needed to resolve ambiguities
   - Examples:
     - "This task mentions X - did you mean A or B?"
     - "What repository should this be done in?"
     - "Are there any specific constraints I should know about?"
   - Continue asking until you have enough clarity to proceed

6. **Execute the task**
   - Once clarified, work on completing the task
   - Keep the user informed of progress

7. **Mark completion**
   - **Success**: Add ` ✓` at the end of the task line
   - **Failure**: Add ` ✗` at the end of the task line, and briefly note why it failed

## TODO.md Format

Tasks are numbered lines. Completed tasks have a tick or cross at the end:

```markdown
1. Implement user authentication ✓
2. Add logging to the data pipeline ✗ (blocked by missing credentials)
3. Refactor the config module
4. Write tests for the new feature
```

In this example, tasks 3 and 4 are available to pick up.

## Example Interactions

### Without task number (error case)
> **User**: /pickup
> **Agent**: Please specify a task number. Usage: `/pickup <task_number>`

### With task number (success case)
> **User**: /pickup 3
> **Agent**: *reads TODO.md*
> Picking up task 3: "Refactor the config module"
>
> A few questions before I start:
> - Which config module are you referring to? (I see config modules in both `catalog-relevance` and `lounge-atlas`)
> - What's the goal of the refactoring - simplification, adding features, or fixing issues?
>
> **User**: The one in catalog-relevance, we want to simplify it
> **Agent**: Got it! I'll refactor the config module in catalog-relevance to simplify it...
> *completes work*
> ✅ Done! I've updated the task in TODO.md with a tick.

### Already completed task
> **User**: /pickup 1
> **Agent**: Task 1 ("Implement user authentication") is already marked as complete (✓). Please choose a different task.

## Important

- **Pick up exactly ONE task** - do not automatically proceed to the next task
- After marking the task complete (✓ or ✗), **stop and await further instructions**
- The user must explicitly invoke pickup again to start the next task

## Notes

- Always ask clarifying questions first - don't assume
- If a task is blocked or impossible, mark it with ✗ and explain why to the user (not in the TODO file)

