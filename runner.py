from datetime import datetime
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence
from uuid import uuid4

from data_model import (
    AlgorithmParameters,
    Demand,
    EvolutionResult,
    Link,
    NaiveResult,
    Network,
)
from evolution import Evolution
from naive_solver import NaiveSolver


@dataclass(frozen=True)
class EvolutionRunnerParams:
    network_name: str
    links: Sequence[Link]
    demands: Sequence[Demand]
    modularities: Sequence[int]
    aggregation: bool
    max_epochs: int
    population_sizes: Sequence[int]
    crossover_probs: Sequence[float]
    tournament_sizes: Sequence[int]
    mutation_probs: Sequence[float]
    mutation_powers: Sequence[float]
    mutation_ranges: Sequence[int]


class EvolutionRunner:
    def __init__(self, params: EvolutionRunnerParams, results_dir: Path):
        self.params = params
        self.results_dir = results_dir

    def run(self):
        for modularity in self.params.modularities:
            evo_network = Network(
                self.params.demands,
                self.params.links,
                modularity,
                self.params.aggregation,
            )
            for parameters in self.parameters_iterator():
                uuid = uuid4()
                evo_result_path = self.results_dir.joinpath(
                    f"{self.params.network_name}-evolution-{uuid}.json"
                )
                evo_result_path.write_text(
                    json.dumps(self.run_evolution(evo_network, parameters))
                )

    def run_evolution(
        self, network: Network, parameters: AlgorithmParameters
    ) -> EvolutionResult:
        evo = Evolution(network, parameters)
        evo.run()
        return evo.get_result()

    def parameters_iterator(self) -> Iterator[AlgorithmParameters]:
        counter = 0
        for mutation_range in self.params.mutation_ranges:
            for mutation_power in self.params.mutation_powers:
                for mutation_prob in self.params.mutation_probs:
                    for tournament_size in self.params.tournament_sizes:
                        for crossover_prob in self.params.crossover_probs:
                            for population_size in self.params.population_sizes:
                                params: AlgorithmParameters = {
                                    "population_size": population_size,
                                    "crossover_prob": crossover_prob,
                                    "tournament_size": tournament_size,
                                    "mutation_prob": mutation_prob,
                                    "mutation_power": mutation_power,
                                    "mutation_range": mutation_range,
                                    "target_fitness": 0,
                                    "max_epochs": self.params.max_epochs,
                                    "stale_epochs_limit": self.params.max_epochs,
                                }
                                counter += 1
                                if counter % 100 == 1:
                                    print(
                                        f"{datetime.now()}: generated {counter} of {self.count_params_combinations()}"
                                    )
                                yield params

    def count_params_combinations(self) -> int:
        return (
            len(self.params.mutation_ranges)
            * len(self.params.mutation_powers)
            * len(self.params.mutation_probs)
            * len(self.params.tournament_sizes)
            * len(self.params.crossover_probs)
            * len(self.params.population_sizes)
        )


@dataclass(frozen=True)
class NaiveRunnerParams:
    network_name: str
    links: Sequence[Link]
    demands: Sequence[Demand]
    modularities: Sequence[int]


class NaiveRunner:
    def __init__(self, params: NaiveRunnerParams, results_dir: Path):
        self.params = params
        self.results_dir = results_dir

    def run(self):
        self.results_dir.mkdir(exist_ok=True)

        for modularity in self.params.modularities:
            uuid = uuid4()
            naive_result_path = self.results_dir.joinpath(
                f"{self.params.network_name}-naive-{uuid}.json"
            )
            naive_network = Network(
                self.params.demands, self.params.links, modularity, True
            )
            naive_result_path.write_text(json.dumps(self.run_naive(naive_network)))

    def run_naive(self, network: Network) -> NaiveResult:
        naive = NaiveSolver(network)
        return naive.solve()
