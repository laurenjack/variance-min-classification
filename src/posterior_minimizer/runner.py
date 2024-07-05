from torch.utils.data import TensorDataset, DataLoader

from src import custom_modules as cm
from src.train import SigmoidBxeTrainer, BoundsAsParam, DirectReg, L1


def run(problem, runs, n, n_test, sizes, hp, first_noisy_index, **kwargs):
    non_zero_count = 0
    # total_weight1 = 0
    for r in range(runs):
        x, y = problem.generate_dataset(n, **kwargs)
        d = x.shape[1]
        x_test, y_test = problem.generate_dataset(n_test, shuffle=True)
        train_set = TensorDataset(x, y)
        train_loader = DataLoader(train_set, hp.batch_size)
        test_set = TensorDataset(x_test, y_test)
        test_loader = DataLoader(test_set, n_test)
        # TODO(Jack) make sizes part of hyperparams
        model = cm.Mlp(sizes, is_bias=False, all_linear=hp.all_linear)
        trainer = SigmoidBxeTrainer()
        l1 = L1(hp.post_constants)
        trainer.run(model, train_loader, test_loader, hp, direct_reg=l1)
        # w1 = abs(model.linears[1].weight[0, 0].item())
        # total_weight1 += w1
        y_shift = y * 2 - 1
        is_non_zero_this_run = False
        for j in range(first_noisy_index, d):
            xj = x[:, j]
            prod = y_shift * xj
            is_same = prod == 1
            same_count = is_same.sum().item()
            is_non_zero_weight = abs(model.linears[0].weight[0, j].item()) > 0.01 and not is_non_zero_this_run
            if is_non_zero_weight:
                non_zero_count += 1
                is_non_zero_this_run = True
        # print(f"Run {r} Non-Zero: {non_zero_count} w1 this run: {w1}")
        print(f"Run {r} Non-Zero: {non_zero_count}")
    print("")
    # print(f"Mean w1: {total_weight1 / runs}")