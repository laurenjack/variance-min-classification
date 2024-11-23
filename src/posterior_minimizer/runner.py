import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from src import custom_modules as cm
from src.posterior_minimizer import regularizer as reg
from src.train import SigmoidBxeTrainer


def run(problem, runs, dp, hp, first_noisy_index, trainer, deviation=None, **kwargs):
    non_zero_count = 0
    ggtr_before = 0
    ggtr_after = 0
    for r in range(runs):
        weight_tracker = None
        # if r == 1:
        #     weight_tracker = wt.AllWeights(len(sizes) - 1)
        max_effective_success, max_grad_before, max_grad_after, zero_state, preds = single_run(problem, dp, hp,
                                                                                  first_noisy_index, trainer,
                                                                                  weight_tracker=weight_tracker, **kwargs)
        # if r == 1:
        #     weight_tracker.show()
        print(max_grad_before)
        print(max_grad_after)
        print(zero_state)
        print(preds)
        if max_grad_before.exceeds_threshold():
            ggtr_before += 1
            if not zero_state.exceeds_threshold():
                print("Gradient greater than reg but zero")
        if max_grad_after.exceeds_threshold():
            ggtr_after += 1
        if zero_state.exceeds_threshold():
            non_zero_count += 1
            if not max_grad_before.exceeds_threshold():
                print("Non zero but gradient less than reg")
        run_report = f"Run {r+1} Non-Zero: {non_zero_count}"
        run_report += f" GGTR before:  {ggtr_before}"
        run_report += f" GGTR ggtr_after:  {ggtr_after}"
        print(run_report)


def single_run(problem, dp, hp, first_noisy_index, trainer, weight_tracker=None,
               print_details=False, **kwargs):
    x, y = problem.generate_dataset(dp.n, **kwargs)
    x_test, y_test = problem.generate_dataset(dp.n_test, shuffle=True)
    train_set = TensorDataset(x, y)
    train_loader = DataLoader(train_set, hp.batch_size)
    test_set = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_set, dp.n_test)
    model = cm.Mlp(hp.sizes, is_bias=False, all_linear=hp.all_linear)

    regularizer = reg.create(dp, hp)
    max_grad_before = regularizer.get_max_gradient(model, x, y)
    max_grad_after = None
    if hp.do_train:
        trainer.run(model, train_loader, test_loader, hp, direct_reg=regularizer,
                                                    weight_tracker=weight_tracker)
        max_grad_after = regularizer.get_max_gradient(model, x, y)
        zero_state = regularizer.get_zero_state(model, x, y)
    else:
        zero_state = reg.NullRegularizationState()

    y_shift = y * 2 - 1
    max_effective_success = 0
    for j in range(first_noisy_index, dp.d):
        xj = x[:, j]
        prod = y_shift * xj
        is_same = prod == 1
        same_count = is_same.sum().item()
        effective_success = abs(same_count - dp.n // 2)
        if effective_success > max_effective_success:
            max_effective_success = effective_success

    sigmoid = nn.Sigmoid()
    a = sigmoid(model(x))
    max_pred = torch.max(a).item()
    min_pred = torch.min(a).item()
    preds = {"max_a": max_pred, "min_a": min_pred}

    if print_details:
        print(model.linears[0].weight)
        print(a[0])

    return max_effective_success, max_grad_before, max_grad_after, zero_state, preds


