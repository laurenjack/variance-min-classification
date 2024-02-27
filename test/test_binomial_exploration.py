import math
import numpy as np

from src import binomial_exploration


def test_get_p_max_equals_x_dict_correct():
    p_no_3s = (1.0 - 0.6**3)**4
    p_max_x_equals_3 = 1.0 - p_no_3s
    p_no_2s_or_3 = (1.0 - 0.6**3 - 3 * 0.6**2*0.4)**4
    p_max_x_equals_2 = 1.0 - p_no_2s_or_3 - p_max_x_equals_3
    p_max_x_equals_0 = 0.4**12
    p_max_x_equals_1 = (0.4**3 + 3 * 0.4**2 * 0.6)**4 - p_max_x_equals_0
    expected_expected_max = p_max_x_equals_1 + 2 * p_max_x_equals_2 + 3 * p_max_x_equals_3
    result = binomial_exploration.get_p_max_equals_x_dict(4,3,p=0.6)
    assert math.isclose(result[3], p_max_x_equals_3, rel_tol=1e-05)
    assert math.isclose(result[2], p_max_x_equals_2, rel_tol=1e-05)
    assert math.isclose(result[1], p_max_x_equals_1, rel_tol=1e-05)
    assert math.isclose(result[0], p_max_x_equals_0, rel_tol=1e-05)
    assert math.isclose(expected_expected_max, binomial_exploration.expected_max_x(4, 3, 0.6), rel_tol=1e-05)
    
def test_single_dim_iterator():
    yt = np.array([
        [1, 1, 1],
        [0, 0, 0],
        [0, 1, 1],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 0, 0]
    ])
    y = yt.transpose()
    result = binomial_exploration.iterate_single_dim(y)
    np.testing.assert_equal(result, np.array([3, 3, 3, 2, 3, 2, 2]))

if __name__ == '__main__':
    test_get_p_max_equals_x_dict_correct()
    test_single_dim_iterator()
    print('All tests passed')