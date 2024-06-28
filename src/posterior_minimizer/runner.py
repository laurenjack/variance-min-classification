from torch.utils.data import TensorDataset, DataLoader

from src import custom_modules as cm
from src.train import SigmoidBxeTrainer, BoundsAsParam, DirectReg, L1


def run(problem, runs, n, n_test, hp, first_noisy_index, deviation, **kwargs):
    over_thresh_count = 0
    non_zero_count = 0
    for r in range(runs):
        x, y = problem.generate_dataset(n, **kwargs)
        d = x.shape[1]
        x_test, y_test = problem.generate_dataset(n_test, shuffle=True)
        train_set = TensorDataset(x, y)
        train_loader = DataLoader(train_set, hp.batch_size)
        test_set = TensorDataset(x_test, y_test)
        test_loader = DataLoader(test_set, n_test)
        # TODO(Jack) make sizes part of hyperparams
        model = cm.Mlp([d, 1], is_bias=False, all_linear=hp.all_linear)
        trainer = SigmoidBxeTrainer()
        l1 = L1(hp.post_constant, hp.single_moving)
        trainer.run(model, train_loader, test_loader, hp, direct_reg=l1)
        y_shift = y * 2 - 1
        is_dev_this_run = False
        is_non_zero_this_run = False
        for j in range(first_noisy_index, d):
            xj = x[:, j]
            prod = y_shift * xj
            is_same = prod == 1
            same_count = is_same.sum().item()
            dev = abs(same_count - n * 0.5)
            does_exceed_x_deviation = dev > deviation and not is_dev_this_run
            if does_exceed_x_deviation:
                over_thresh_count += 1
                is_dev_this_run = True
            is_non_zero_weight = abs(model.linears[0].weight[0, j].item()) > 0.01 and not is_non_zero_this_run
            if is_non_zero_weight:
                non_zero_count += 1
                is_non_zero_this_run = True
            if is_non_zero_weight and not does_exceed_x_deviation:
                print('Warning, Deviation was not exceeded but weight was still zero')
                print(f'Same count: {same_count}')
                print('')
        print(f"Run {r} Non-Zero: {non_zero_count} Over threshold: {over_thresh_count}")