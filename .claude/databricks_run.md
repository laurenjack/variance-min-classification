# Databricks Notebook Run

This command helps launch a Databricks notebook job using the `run_job.sh` script in the `catalog-relevance` repository.

## Arguments

The argument passed to this command is: $ARGUMENTS

If arguments are provided, parse them to extract the notebook path and any pre-specified parameters.

## Process

1. **Determine the notebook path:**
   - If provided in arguments, use that
   - Otherwise, ask the user for the notebook path (relative to `notebooks/` directory, e.g., `two_tower/customer_initial` or `manual_analysis/count_datasets`)

2. **Read the notebook file** at `catalog-relevance/notebooks/<path>.py` to identify all widget parameters defined via `dbutils.widgets.text()` calls. Extract:
   - Parameter name
   - Default value

3. **Categorize the parameters:**
   - **Date parameters**: `run_date`, `start_date`, `end_date` (handled specially by the script)
   - **Branch parameter**: `branch_name` (automatically set by the script, skip this)
   - **Extra parameters**: All other widget parameters

4. **Ask the user for:**
   - **Date(s)**: Based on whether notebook uses `run_date` or `start_date`/`end_date`
   - **Number of workers**: How many cluster workers (used for both min and max for fixed sizing)
   - **Instance type** (optional): Default is `rgd-fleet.16xlarge`, options include `rgd-fleet.8xlarge`, `rgd-fleet.4xlarge`, `rgd-fleet.2xlarge`
   - **Extra parameters**: Show the user which extra parameters exist with their defaults, ask if they want to override any

5. **Build and run the job:**
   - Construct the extra params JSON from any non-default values: `{"param1": "value1", "param2": "value2"}`
   - The script has TWO MODES based on whether the notebook uses `run_date` or `start_date`/`end_date`:

   **For notebooks with `run_date` (single date mode):**
   ```bash
   cd /Users/jlaurenson/projects/catalog-relevance && ./scripts/manual_job_runs/run_job.sh \
     <notebook_path> \
     <run_date> \
     "<job_name>" \
     <min_workers> <max_workers> \
     <instance_type>
   ```

   **For notebooks with `start_date`/`end_date` (date range mode):**
   ```bash
   cd /Users/jlaurenson/projects/catalog-relevance && ./scripts/manual_job_runs/run_job.sh \
     <notebook_path> \
     <start_date> <end_date> \
     "<job_name>" \
     <min_workers> <max_workers> \
     <instance_type>
   ```

   **IMPORTANT:** The script detects the mode by checking if the 3rd argument is a date (YYYY-MM-DD format):
   - If 3rd arg IS a date → date range mode → notebook receives `start_date` and `end_date`
   - If 3rd arg is NOT a date → single date mode → notebook receives `run_date`

   **Extra parameters:** If you need to pass extra parameters, omit them to use the default `{}`, or pass valid JSON as the last argument (without quotes around the JSON object itself).

6. **Report the Databricks run URL** from the output so the user can monitor the job.

## Example Interaction (Single Date - run_date)

> **User**: /databricks_run two_tower/fdna_customer_average
> **Agent**: I found `notebooks/two_tower/fdna_customer_average.py`. It has these widget parameters:
>
> **Date parameter:**
> - `run_date` (default: 2025-01-01)
>
> **Extra parameters:**
> - `alpha` (default: 1.0)
>
> What date would you like to run? How many workers and what instance type? Do you want to override `alpha`?
>
> **User**: 2026-01-01, 4 workers on 4xlarge, keep alpha default
> **Agent**: Running the job...
> *executes:* `./run_job.sh two_tower/fdna_customer_average 2026-01-01 "fdna_customer_average_2026-01-01" 4 4 rgd-fleet.4xlarge`
>
> Job submitted! Run URL: https://zalando-e2.cloud.databricks.com/#job/runs/12345

## Example Interaction (Date Range - start_date/end_date)

> **User**: /databricks_run manual_analysis/count_datasets
> **Agent**: I found the notebook with these parameters:
>
> **Date parameters:**
> - `start_date` (default: 2026-01-01)
> - `end_date` (default: 2026-01-01)
>
> **Extra parameters:**
> - `datasets` (default: FDNA_CUSTOMER_AVERAGE,TOPK_TP4U_ATTRIBUTION)
>
> What dates and configuration would you like?
>
> **User**: 2026-01-01 to 2026-01-03, 2 workers 4xlarge, datasets should be TWO_TOWER_CUSTOMER
> **Agent**: Running with extra params: `{"datasets": "TWO_TOWER_CUSTOMER"}`
> *executes:* `./run_job.sh manual_analysis/count_datasets 2026-01-01 2026-01-03 "count_datasets_2026-01-01" 2 2 rgd-fleet.4xlarge '{"datasets": "TWO_TOWER_CUSTOMER"}'`

## Notes

- The script automatically deploys the current git branch to Databricks before running
- For fixed worker count, set min_workers = max_workers
- The script passes `branch_name` automatically based on the current git branch
- Extra parameters are passed as a JSON object and merged with the date parameters
- Only include extra parameters in the JSON if the user wants to override the default
- **Critical:** Match the date mode to what the notebook expects - using the wrong mode will pass incorrect parameter names to the notebook
