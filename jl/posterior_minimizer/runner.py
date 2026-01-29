import torch
from torch import nn

import jl.posterior_minimizer.empirical.reporter
from jl.posterior_minimizer.empirical import modules
from jl.posterior_minimizer import custom_modules as cm
from jl.posterior_minimizer.empirical.reporter import show_greatest_grad
from jl.posterior_minimizer.hyper_parameters import DataParameters, HyperParameters
from jl.posterior_minimizer import regularizer as reg, weight_tracker as wt, variance as v

def create_weight_tracker(variance: v.Variance, dp: DataParameters, hp: HyperParameters):
    if hp.weight_tracker_type:
        num_layers = len(hp.sizes) - 1
        if hp.weight_tracker_type == 'Weight':
            return wt.AllWeights(num_layers, node_limit=10)
        if hp.weight_tracker_type == 'Gradient':
            return wt.GradAtZeroTracker(num_layers, dp, hp, variance, node_limit=10)
        raise ValueError(f'Unknown Weight Tracker Type {hp.weight_tracker_type}')
    return wt.WeightTracker()


def run(problem, runs, dp, hp, trainer, **kwargs):
    non_zero_count = 0
    ggtr_before = 0
    ggtr_after = 0
    for r in range(runs):
        max_grad_before, max_grad_after, zero_state, preds = single_run(problem, dp, hp,
                                                                                  trainer,
                                                                                  **kwargs)
        print(max_grad_before)
        print(max_grad_after)
        print(zero_state)
        print(preds)
        if max_grad_before.exceeds_threshold():
            ggtr_before += 1
            if not zero_state.exceeds_threshold():
                print("Gradient greater than reg but zero")
        if hp.do_train and max_grad_after.exceeds_threshold():
            ggtr_after += 1
        if zero_state.exceeds_threshold():
            non_zero_count += 1
            if not max_grad_before.exceeds_threshold():
                print("Non zero but gradient less than reg")
        run_report = f"Run {r+1} Non-Zero: {non_zero_count}"
        run_report += f" GGTR before:  {ggtr_before}"
        run_report += f" GGTR ggtr_after:  {ggtr_after}"
        print(run_report)


def single_run(problem, dp, hp, trainer,
               print_details=False, **kwargs):
    x, y, _ = problem.generate_dataset(dp.n, **kwargs)
    x_test, y_test, _ = problem.generate_dataset(dp.n_test, shuffle=True)
    model = modules.Mlp(dp, hp)
    # if hp.implementation == 'new':
    #     model = modules.Mlp(dp, hp)
    # else:
    #     model = cm.Mlp(hp.sizes, is_bias=hp.is_bias, all_linear=hp.all_linear)
    # linear = model.linears[0]
    # b = (2 * 3 / 1) ** 0.5
    # new_weights = torch.empty_like(linear.weight).uniform_(-b, b)
    # linear.weight.data = new_weights
    max_grad_before = None
    max_grad_after = None
    zero_state = None
    has_noisy_d = dp.d - dp.true_d > 0
    variance = v.create_variance(dp, hp)
    if hp.implementation == 'new':
        regularizer = None
    else:
        regularizer = reg.create(variance, dp, hp)
    weight_tracker = create_weight_tracker(variance, dp, hp)
    if has_noisy_d:
        if hp.implementation == 'new':
            max_grad_before = show_greatest_grad(model, x, y, dp, hp)
        else:
            max_grad_before = regularizer.get_max_gradient(model, x, y, true_d=dp.true_d)

    if hp.do_train:
        trainer.run(model, x, y, x_test, y_test, dp, hp, direct_reg=regularizer,
                                                    weight_tracker=weight_tracker)
        if has_noisy_d:
            if hp.implementation == 'new':
                max_grad_after = show_greatest_grad(model, x, y, dp, hp)
                zero_state = reg.get_zero_state(model, x, y, 0.0001, true_d=dp.true_d)
            else:
                max_grad_after = regularizer.get_max_gradient(model, x, y, true_d=dp.true_d)
                zero_state = regularizer.get_zero_state(model, x, y, true_d=dp.true_d)
    else:
        zero_state = src.empirical.reporter.NullRegularizationState()

    sigmoid = nn.Sigmoid()
    a = sigmoid(model(x))
    mean_confidence = torch.mean(torch.max(a, 1 - a)).item()
    print(f"Mean confidence: {mean_confidence}")
    max_pred = torch.max(a).item()
    min_pred = torch.min(a).item()
    preds = {"max_a": max_pred, "min_a": min_pred}

    if print_details:
        print('Layer 0')
        print(model.linears[0].weight[:, 0:2])
        print('Layer 1')
        print(model.linears[1].weight)
        # print(a[0])

    if weight_tracker:
        weight_tracker.show()

    return max_grad_before, max_grad_after, zero_state, preds


def just_give_w(x, y, dp, hp, trainer):
    model = cm.Mlp(hp.sizes, is_bias=False, all_linear=hp.all_linear)
    regularizer = reg.create(dp, hp)
    trainer.run(model, x, y, x, y, dp, hp, direct_reg=regularizer)
    return model.linears[0].weight[0, 0]


