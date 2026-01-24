# Check Dataset Row Counts

This command is a shortcut to run the `count_datasets` notebook for checking row counts of datasets from the registry.

## Arguments

The argument passed to this command is: $ARGUMENTS

## Process

Simply invoke `/databricks_run manual_analysis/count_datasets` which will:
1. Read the notebook and show available parameters
2. Ask for dates and the `datasets` parameter (comma-separated list of registry enum names)
3. Run the job and return the Databricks URL

## Quick Reference

The `datasets` parameter accepts comma-separated enum names from `catalog_relevance/datasets/registry.py`:

**Two Tower / Top-K:**
- `TWO_TOWER_CUSTOMER` - Customer lounge actions
- `FDNA_CUSTOMER_AVERAGE` - Customer fDNA weighted averages
- `TOPK_TP4U_ATTRIBUTION` - TP4U attributed requests
- `TOPK_TP4U_FEATURE_ENRICHED` - Feature-enriched TP4U requests

**Attribution:**
- `STUDIO_CORE_CONFIG_ATTRIBUTION` - Config attribution data

**Feature Store:**
- `CUSTOMER_FS_ADD_TO_CART_FEATURES` - Customer add-to-cart features

## Example

> **User**: /check_rows
> **Agent**: *invokes /databricks_run manual_analysis/count_datasets*
