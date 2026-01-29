from jl.results.simple_mlp.simple_mlp_conf import run_experiment


def main():
    width_range = list(range(80, 6001, 20))
    num_runs = 1
    run_experiment(width_range, num_runs)


if __name__ == "__main__":
    main()
