import torch

from src import dataset_creator

runs = 10000
n = 100
d = 10

cov_sum = (1 + (d - 1) / n)
total_var = cov_sum / n
total_sd = total_var ** 0.5

expected_cov = torch.tensor([[1.0, 1 / n], [1 / n, 1.0]])

print(expected_cov)
print(total_var)


# all_noise = dataset_creator.AllNoise(num_class=2, d=d)
#
# mean_cov_mat = torch.zeros(d, d)
# mean_ols_magnitude = torch.zeros(d)
# for r in range(runs):
#     x, y = all_noise.generate_dataset(n, shuffle=True)
#     y_shift = y * -2.0 + 1
#     cov_mat = x.t() @ x
#     ols = torch.inverse(cov_mat) @ (x.t() @ y_shift)
#     mean_cov_mat += (torch.abs(cov_mat) / n) ** 2
#     mean_ols_magnitude += ols ** 2
#
# print((mean_cov_mat / runs) ** 0.5)
# print((mean_ols_magnitude/ runs) ** 0.5)
