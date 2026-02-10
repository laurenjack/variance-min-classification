# Lambda Train

This command pushes changes to git and then runs reward model training on a Lambda Labs instance.

## Inputs

- `instance_ip` (required): IP address of the running Lambda Labs instance
- `learning_rate` (optional): Learning rate override (e.g. 3e-5)
- `warmup_steps` (optional): Number of quadratic warmup steps (default: 30)

## Process

1. **Push changes to git**
   - Run the `/push` command to stage, commit, and push all changes
   - Wait for the push to complete successfully before proceeding

2. **Run training on the Lambda instance**
   - Run `infra/lambda_train.sh` with the provided IP and `--background` mode:
     ```bash
     ./infra/lambda_train.sh <instance_ip> --background
     ```
   - If a `learning_rate` was provided, include it:
     ```bash
     ./infra/lambda_train.sh <instance_ip> --background --learning-rate <learning_rate>
     ```
   - If `warmup_steps` was provided, include it:
     ```bash
     ./infra/lambda_train.sh <instance_ip> --background --warmup-steps <warmup_steps>
     ```
   - This will:
     - SSH into the instance
     - Clone or pull the latest code from git
     - Set up the Python environment and install dependencies
     - Start training in the background with `nohup`

## Prerequisites

- Environment variables loaded from `.env` (run `source .env`)
- A running Lambda Labs instance (use `infra/lambda_launch.sh` to start one)
- `HF_TOKEN` environment variable set (for gated models like Llama)
- `LAMBDA_SSH_KEY_PATH` environment variable set
- Changes committed and ready to push

## Monitoring

After training starts, monitor with:
```bash
ssh -i $LAMBDA_SSH_KEY_PATH ubuntu@<instance_ip> 'tail -f ~/variance-min-classification/training.log'
```

## Next Steps

- Download results: `./infra/lambda_download.sh <instance_ip>`
- Terminate instance: `./infra/lambda_terminate.sh`
