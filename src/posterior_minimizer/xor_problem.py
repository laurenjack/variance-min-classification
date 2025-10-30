from src import hyper_parameters
from src import dataset_creator, train
from src.posterior_minimizer import runner

# torch.manual_seed(3991)  # 3962

n = 100
n_test = 100
percent_correct = 0.8
true_d = 2
noisy_d = 20
d = true_d + noisy_d

dp = hyper_parameters.DataParameters(percent_correct, n, n_test, d, true_d=true_d)

hp = hyper_parameters.HyperParameters(batch_size=n,
                                      epochs=200,
                                      learning_rate=0.01,  # 1.0 / d ** 0.5,
                                      momentum=0.9,
                                      weight_decay=0.001,
                                      desired_success_rate=0.5,
                                      sizes=[d, 12, 12, 1],
                                      do_train=True,
                                      gamma=1.0,
                                      is_adam=True,
                                      all_linear=False,
                                      is_bias=False,
                                      relu_bound=0.05,
                                      reg_type="NoReg",
                                      weight_tracker_type='Weight',
                                      implementation='new',
                                      reg_epsilon=0.0,
                                      print_epoch=True,
                                      print_batch=False)


problem = dataset_creator.HyperXorNormal(true_d, percent_correct, noisy_d)
trainer = train.SigmoidBxeTrainer()

max_grad_before, max_grad_after, zero_state, preds = runner.single_run(problem, dp, hp, trainer, print_details=False, shuffle=False)
print(max_grad_before)
print(max_grad_after)
print(zero_state)
print(preds)

# print(weight_tracker.layers[0][-1])

