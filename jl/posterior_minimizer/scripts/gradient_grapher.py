import numpy as np
import matplotlib.pyplot as plt

from jl.posterior_minimizer import hyper_parameters, dataset_creator, train
from jl.posterior_minimizer import runner

def g1(w1, x, y):
    n = x.shape[0]
    x1 = x[:, 1]
    a = 1 / (1 + np.exp(-w1 * x1))
    return np.sum((y - a) * x1) / n

def g2(w0, w1, x, y):
    n = x.shape[0]
    x0 = x[:, 0]
    x1 = x[:, 1]
    a = 1 / (1 + np.exp(-(w0 * x0 + w1 * x1)))
    return np.sum((y - a) * x1) / n


def plot_gradient(w0, x, y):
    w_range = np.linspace(-1.0, 1.0, 500)
    dm_w0 = 2 * np.mean((2 * y - 1) * x[:, 0])
    print("Direct Mean w0 =", dm_w0)

    g1_values = [g1(w1, x, y) for w1 in w_range]
    g2_values = [g2(w0, w1, x, y) for w1 in w_range]

    # Plot g1 first (in blue) and then g2 (in green)
    plt.figure(figsize=(8, 6))
    plt.plot(w_range, g1_values, label='g1(w)', color='blue')
    plt.plot(w_range, g2_values, label='g2(w)', color='green')
    plt.axhline(0, color='red', linestyle='--', linewidth=1, label='y = 0')
    plt.title('Comparison of g1(w) and g2(w)')
    plt.xlabel('w')
    plt.ylabel('Gradient Value')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    n = 100
    problem = dataset_creator.MultivariateNormal(true_d=1,
                                                 percent_correct=0.75,
                                                 noisy_d=1)
    x, y = problem.generate_dataset(n, shuffle=True)

    n_test = 10
    percent_correct = 0.75
    true_d = 1
    noisy_d = 0
    d = true_d + noisy_d

    dp = hyper_parameters.DataParameters(percent_correct, n, n_test, d)

    hp = hyper_parameters.HyperParameters(batch_size=n // 4,
                                          epochs=100,
                                          learning_rate=0.1,  # 1.0 / d ** 0.5,
                                          momentum=0.9,
                                          weight_decay=0.00,
                                          desired_success_rate=0.5,
                                          sizes=[d, 1],
                                          do_train=True,
                                          gamma=1.0,
                                          is_adam=False,
                                          all_linear=False,
                                          reg_type="NoReg",
                                          var_type="Analytical",
                                          reg_epsilon=0.0,
                                          print_epoch=False,
                                          print_batch=False)

    trainer = train.SigmoidBxeTrainer()
    x0 = x[:, 0].view(n, 1)
    w0 = runner.just_give_w(x0, y, dp, hp, trainer).item()
    print(f'Opt w0: {w0}')
    plot_gradient(w0, x.numpy(), y.numpy())
