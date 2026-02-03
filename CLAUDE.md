# CLAUDE.md

Do not commit unless asked to.

## Environment

Activate the Python virtual environment before running any code:

```bash
source venv/bin/activate
```

When running tests or code, always start the venv as part of the same command:
```bash
source venv/bin/activate && python -m pytest tests/
```

When you make a change to a test or code module, run ALL tests affected by that change.

## Project Structure

This project explores variance-minimizing classification models with various architectures and regularization strategies.

- `jl/config.py` - `Config` class holding all model/training hyperparameters with validation
- `jl/models.py` - Model architectures: `Resnet`, `MLP`, `MultiLinear`, `KPolynomial`, `SimpleMLP`
- `jl/model_creator.py` - Factory functions that instantiate models from a `Config`
- `jl/single_runner.py` - `train_once()` training loop (handles optimizers, schedulers, scaled regularization, weight tracking)
- `jl/ten_runner.py` - Multi-run training (`train_multi`) for ensemble experiments
- `jl/feature_experiments/` - Experiment scripts and specialized modules (dropout, scaled regularization, optimizers)
- `jl/posterior_minimizer/` - Posterior/variance analysis and weight tracking utilities
- `jl/reward_model/` - RLHF reward model training (HuggingFace transformers)

## Running Experiments

```bash
source venv/bin/activate
python -m jl.feature_experiments.run_single_feature
python -m jl.feature_experiments.run_feature_combinations
```

## Reward Model Training

### Local
```bash
source venv/bin/activate
python -m jl.reward_model.main
```

### SageMaker
First set up AWS credentials and infrastructure:
```bash
source ~/.cursor_bootstrap.sh
./infra/sagemaker-setup.sh
```

Then launch training:
```bash
source ~/.cursor_bootstrap.sh && source venv/bin/activate
python -m jl.reward_model.launch_sagemaker
```

Note: Requires SageMaker quota for `ml.g6e.xlarge` in eu-central-1. Request via AWS Service Quotas if needed.
