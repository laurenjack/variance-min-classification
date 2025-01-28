import torch

from src import hyper_parameters
from src import dataset_creator, train
from src.posterior_minimizer import runner


# torch.manual_seed(3956) # 3917

runs = 10000
n = 100
n_test = 10
percent_correct = 0.8
true_d = 1
noisy_d = 3
d = true_d + noisy_d

dp = hyper_parameters.DataParameters(percent_correct, n, n_test, d, true_d=true_d)

hp = hyper_parameters.HyperParameters(batch_size=n // 4,
                                      epochs=300,
                                      learning_rate=0.1,  # 1.0 / d ** 0.5,
                                      momentum=0.9,
                                      weight_decay=0.00,
                                      desired_success_rate=0.5,
                                      sizes=[d, 3, 1],
                                      do_train=False,
                                      gamma=1.0,
                                      is_adam=False,
                                      all_linear=True,
                                      reg_type="L1",
                                      var_type="Analytical",
                                      reg_epsilon=0.0,
                                      print_epoch=False,
                                      print_batch=False)


problem = dataset_creator.MultivariateNormal(true_d, percent_correct, noisy_d)
trainer = train.SigmoidBxeTrainer()
runner.run(problem, runs, dp, hp, trainer, shuffle=True)
