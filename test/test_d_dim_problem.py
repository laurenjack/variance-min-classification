import math
import numpy as np

from src import d_dim_problem

def test_count_segmentations():
    # x = np.array([[-1.0, -1.0], [-0.9, 1.0], [1.0, -1.0], [0.9, 1.0]])
    # result = d_dim_problem.count_segmentations(x)
    # analytical_result = d_dim_problem.analytic_segmentations(4, 2)
    #
    # assert result == 14 == analytical_result
    #
    # x = np.array([[-1.0, -1.0], [-0.9, 1.0], [1.0, -1.0], [0.9, 1.0], [0.1, 0.8]])
    # result = d_dim_problem.count_segmentations(x)
    # analytical_result = d_dim_problem.analytic_segmentations(5, 2)
    # assert result == 22 == analytical_result
    #
    # analytical_result = d_dim_problem.analytic_segmentations(6, 2)
    # for i in range(10):
    #     x = np.random.randn(6, 2)
    #     assert d_dim_problem.count_segmentations(x) == 32 == analytical_result

    for d in range(2, 9):
        for n in range(d, 9):
            analytical_result = d_dim_problem.analytic_segmentations(n, d)
            comb = math.comb(n, d)
            new_bound = d_dim_problem.new_bound(n, d)
            other_bound = d_dim_problem.new_bound(n, d-1)
            # print(n, d)
            # print(analytical_result)
            for j in range(1):
                # x = []
                # for k in range(d):
                #     xi = np.arange(n)
                #     np.random.shuffle(xi)
                #     xi = 5 * (xi + np.random.uniform(0.25, 0.75) - n / 2) / n
                #     x.append(xi)
                # x = np.array(x).transpose()
                x = np.random.randn(n, d) # .astype(np.float32)
                # empirical_result, er2, min_out_of_plane, max_in_plane, edges = d_dim_problem.count_segmentations(x)
                true_result = d_dim_problem.count_segmentations_new(x)
                # if empirical_result < comb:
                print(n, d)
                print(comb)
                # print(empirical_result)
                print(true_result)
                # print(er2)
                print(new_bound)
                # print(other_bound)
                # print(analytical_result)
                # print(min_out_of_plane)
                # print(max_in_plane)
                # print(edges)
                print('')
                # assert empirical_result >= comb



if __name__ == '__main__':
    test_count_segmentations()
    print("All tests passed!")