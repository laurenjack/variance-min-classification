import torch

from src import hyper_parameters
from src import dataset_creator, train
from src.posterior_minimizer import runner


torch.manual_seed(3958) # 3917

runs = 100
n = 100
n_test = 10
percent_correct = 0.8
true_d = 1
noisy_d = 30
d = true_d + noisy_d

dp = hyper_parameters.DataParameters(percent_correct, n, n_test, d, true_d=true_d)

hp = hyper_parameters.HyperParameters(batch_size=n,
                                      epochs=300,
                                      learning_rate=0.3,  # 1.0 / d ** 0.5,
                                      momentum=0.0,
                                      weight_decay=0.0,
                                      desired_success_rate=0.5,
                                      sizes=[d, 1],  # 40, 30,
                                      do_train=True,
                                      gamma=1.0,
                                      is_adam=True,
                                      all_linear=True,
                                      is_bias=False,
                                      reg_type="L1",
                                      implementation='old',
                                      reg_epsilon=0.0,
                                      print_epoch=False,
                                      print_batch=False)


problem = dataset_creator.MultivariateNormal(true_d, percent_correct, noisy_d)
trainer = train.SigmoidBxeTrainer()
runner.run(problem, runs, dp, hp, trainer, shuffle=True)
