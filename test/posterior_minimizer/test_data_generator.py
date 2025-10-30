from jl.posterior_minimizer import data_generator


def test_uniform_one_true_dim():
    x, y = data_generator.uniform_one_true_dim(1000, 3, 1.0)
    num_correct_sign = _get_num_correct_sign(x, y, 2000, 3)
    assert num_correct_sign[0] == 2000
    assert 900 < num_correct_sign[1] < 1100
    assert 900 < num_correct_sign[2] < 1100

    x, y = data_generator.uniform_one_true_dim(1000, 2, 0.8)
    num_correct_sign = _get_num_correct_sign(x, y, 2000, 2)
    assert 1500 < num_correct_sign[0] < 1700
    assert 900 < num_correct_sign[1] < 1100

    x, y = data_generator.uniform_one_true_dim(1000, 1, 0.5)
    num_correct_sign = _get_num_correct_sign(x, y, 2000, 1)
    assert 900 < num_correct_sign[0] < 1100


def _get_num_correct_sign(x, y, expected_n, expected_d):
    assert x.shape[0] == expected_n
    assert x.shape[1] == expected_d
    assert y.shape[0] == expected_n
    assert y.sum().item() == expected_n / 2

    num_correct_sign = [0] * expected_d
    for i in range(expected_n):
        for j in range(expected_d):
            if y[i].item() == 1 and x[i, j].item() > 0 or y[i].item() == 0 and x[i, j].item() < 0:
                num_correct_sign[j] += 1
    return num_correct_sign


if __name__ == '__main__':
    test_uniform_one_true_dim()
    print('All tests passed!')