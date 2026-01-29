from jl.results.simple_mlp.simple_mlp_conf import run_experiment


def main():
    width_range = list(range(3, 81, 1))
    num_runs = 20
    run_experiment(width_range, num_runs)


if __name__ == "__main__":
    main()
