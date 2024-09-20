import torch
from scipy.stats import binom, norm

from src import hyper_parameters, train

from src import dataset_creator
from src.posterior_minimizer import runner, regularizer as reg

torch.manual_seed(56433)

runs = 100
n = 100
n_test = 100
d = 40
sizes = [d, 30, 20, 10, 1]

percent_correct = 0.5
desired_success_rate = 0.72
num_nodes = sum(sizes[:-1])

bp_single = (1 - desired_success_rate ** (1 / num_nodes)) / 2
print(f'bp: {bp_single}')
threshold_success = binom.ppf(1 - bp_single, n, 0.5) # - 1
effective_bp = 1 - binom.cdf(threshold_success, n, 0.5)
print(f'effective bp: {effective_bp}')
effective_success_rate = (1 - 2 * effective_bp) ** num_nodes
print(f"Effective success rate: {effective_success_rate}")
deviation = threshold_success - n * 0.5
print(deviation)
binom_post_constant = threshold_success / n - 0.5
# post_constant = (prop_product / (n / z ** 2 + 1)) ** 0.5
print(f"Binom Post Constant: {binom_post_constant}")

dp = hyper_parameters.DataParameters(percent_correct, n, n_test, d)

hp = hyper_parameters.HyperParameters(batch_size=n,
                                      epochs=400,
                                      learning_rate= 1.0 / d ** 0.5,
                                      momentum=0.0,
                                      weight_decay=0.0,
                                      sizes=sizes,
                                      desired_success_rate=desired_success_rate,
                                      is_adam=True,
                                      all_linear=True,
                                      reg_type="L1",
                                      gamma=0.85,
                                      reg_epsilon=0.001,
                                      print_epoch=False,
                                      print_batch=False)

problem = dataset_creator.AllNoise(num_class=2, d=d)
trainer = train.SigmoidBxeTrainer()
runner.run(problem, runs, dp, hp, 0, trainer, deviation=deviation, shuffle=True)

