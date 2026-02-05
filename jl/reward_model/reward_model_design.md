In this part of the repo, we demonstrate that despite an increasing validation loss, double descent occurs in the win-rate when training LLM based reward models, as predicted by our theory. That is, we show a superior win rate can be obtained by training reward models past the interpolation threshold, pushing the training probabilities output by the model to 1.

For now, the code supports  small models <10B parameters from the Llama family and training the reward model on the Anthropic Helpful-Harmless dataset (Anthropic/hh-rlhf). However, we will expand to other datasets

# Evaluation 

For now, we'll evaluate on the Anthropic/hh-rlhf holdout set, as a proof of concept. We'll come back to more advanced evaluations

# System Design

Currently, the model trains on a single GPU which takes too long if we are to train for ~50 epochs. As such we'll switch to an 8 GPU instance, with a copy of the model on each GPU. That way, we can increase the batch_size to 256

# Hyperparameter Search

Training a reward model into the double descent regime is uncharted territory, it is likely the case that the hyperparameters are tuned to catiously to stop overfitting, and to keep the rewards in a sensible range. We are not constrained in the same way here.

So when it comes to  hyperparameters, we want to speed up training significantly, we should experiment with higher learning rates amongst other things. We will start with these values (dervied from Tülu 3 - Lambert et al 2024 and Skywork-Reward-V2 Liu et al 2025):
learning rate: 3 × 10-6
Gradient Norm Threshold: 1.0
batch size: 256
epochs: 1
warm up period 20%

We'll start with this low learning rate and explore the stability, first we'll increase it in orders of magnititude of 10. The key quanties we wish to observe are the training loss and training win rate, we want to see these increasing as fast as possible while remaining stable. So for us that will be 3 × 10-5, and 3 × 10-4

We'll use our sagemaker logging to capture this. We'll make the logging more structured, so that another agent may pull the logs, write them it into a file, and graph it. We'll  need to write the bash script that retrieve the logs, writes them to a file, and also write the python script that will graph it (and call it from our script). We can use the data directory to hold the data regarding the training metrics.