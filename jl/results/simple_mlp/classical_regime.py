from jl.multi_experiment_grapher import GraphConfig
from jl.results.simple_mlp.simple_mlp_conf import run_experiment


def main():
    width_range = list(range(3, 81, 1))
    num_runs = 20
    graph_config = GraphConfig(show_validation_loss=True)
    run_experiment(width_range, num_runs, graph_config)


if __name__ == "__main__":
    main()
