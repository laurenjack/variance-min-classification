from jl.multi_experiment_grapher import GraphConfig
from jl.results.resnet.resnet_conf import run_experiment


def main():
    width_range = list(range(80, 4001, 40))
    num_runs = 1
    graph_config = GraphConfig(
        # constant_name="Classical Minimum",
        # constant_value=0.3,
        show_validation_loss=True,
        show_line=False,
    )
    run_experiment(width_range, num_runs, graph_config)


if __name__ == "__main__":
    main()
