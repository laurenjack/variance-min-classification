# CLAUDE.md

Do not commit unless asked to.

## Environment

Activate the Python virtual environment before running any code:

```bash
source venv/bin/activate
```

## Project Structure

This project explores variance-minimizing classification models with various architectures and regularization strategies.

- `jl/config.py` - `Config` class holding all model/training hyperparameters with validation
- `jl/models.py` - Model architectures: `Resnet`, `MLP`, `MultiLinear`, `KPolynomial`, `SimpleMLP`
- `jl/model_creator.py` - Factory functions that instantiate models from a `Config`
- `jl/single_runner.py` - `train_once()` training loop (handles optimizers, schedulers, scaled regularization, weight tracking)
- `jl/ten_runner.py` - Multi-run training (`train_multi`) for ensemble experiments
- `jl/feature_experiments/` - Experiment scripts and specialized modules (dropout, scaled regularization, optimizers)
- `jl/posterior_minimizer/` - Posterior/variance analysis and weight tracking utilities

## Running Experiments

```bash
source venv/bin/activate
python -m jl.feature_experiments.run_single_feature
python -m jl.feature_experiments.run_feature_combinations
```
