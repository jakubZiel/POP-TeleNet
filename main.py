import multiprocessing
from datetime import datetime
from pathlib import Path
from typing import Sequence, Tuple

from data_model import Demand, Link
from parsing import NetworkParser
from runner import (
    EvolutionRunner,
    EvolutionRunnerParams,
    NaiveRunner,
    NaiveRunnerParams,
)

_RESULTS_PATH = Path("results")

_MODULARITIES = [1, 10, 50]


def generate_aggregation_one_repeat(args: Tuple[str, Sequence[Demand], Sequence[Link]]):
    runner_params = EvolutionRunnerParams(
        network_name=args[0],
        links=args[2],
        demands=args[1],
        modularities=_MODULARITIES,
        aggregation=True,
        max_epochs=1000,
        population_sizes=[10, 40, 100],
        crossover_probs=[0.0, 0.2, 0.5, 1.0],
        tournament_sizes=[1, 2, 4, 8],
        mutation_probs=[0.1, 0.25, 0.5, 1.0],
        mutation_powers=[1],
        mutation_ranges=[1, 2, 4, 8],
    )
    EvolutionRunner(runner_params, _RESULTS_PATH).run()


def generate_no_aggregation_one_repeat(
    args: Tuple[str, Sequence[Demand], Sequence[Link]]
):
    runner_params = EvolutionRunnerParams(
        network_name=args[0],
        links=args[2],
        demands=args[1],
        modularities=_MODULARITIES,
        aggregation=False,
        max_epochs=1000,
        population_sizes=[10, 40, 100],
        crossover_probs=[0.0, 0.2, 0.5],
        tournament_sizes=[1, 3, 9],
        mutation_probs=[0.1, 0.5, 1.0],
        mutation_powers=[1],
        mutation_ranges=[1, 3],
    )
    EvolutionRunner(runner_params, _RESULTS_PATH).run()


def generate_naive(network_name: str, demands: Sequence[Demand], links: Sequence[Link]):
    naive_params = NaiveRunnerParams(
        network_name=network_name,
        links=links,
        demands=demands,
        modularities=_MODULARITIES,
    )
    NaiveRunner(naive_params, _RESULTS_PATH).run()


def generate(network_name: str, network_path: Path, repeats: int):
    parser = NetworkParser(network_path)
    demands = parser.demands()
    links = parser.links()
    # naive
    generate_naive(network_name, demands, links)
    # evolution aggregation
    pool = multiprocessing.Pool(processes=repeats)
    pool.map(
        generate_aggregation_one_repeat,
        [(network_name, demands, links) for _ in range(repeats)],
    )
    # evolution no aggregation
    pool.map(
        generate_no_aggregation_one_repeat,
        [(network_name, demands, links) for _ in range(repeats)],
    )


def main():
    print(f"start time {datetime.now()}")
    generate(
        network_name="polska", network_path=Path("networks/polska.xml"), repeats=10
    )
    print(f"end time {datetime.now()}")


if __name__ == "__main__":
    main()
