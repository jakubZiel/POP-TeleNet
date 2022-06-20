import math
import sys
from collections.abc import Sequence
from random import randint, random
from typing import Dict, List, Tuple

import numpy as np

from data_model import (
    AdmissablePath,
    AlgorithmParameters,
    Demand,
    EvolutionResult,
    LinkResult,
    Network,
    Specimen,
    SpecimenDemand,
)


class Evolution:
    def __init__(self, network: Network, parameters: AlgorithmParameters):
        self.demands = {demand.id: demand for demand in network.demands}
        self.links = {link.id: link for link in network.links}
        self.modularity = network.modularity
        self.aggregation = network.aggregation

        self.params = parameters

        self.population: List[Specimen] = []
        self.best_specimen = Specimen([], sys.maxsize)
        self.prev_best_fitness = math.inf
        self.current_generation = 0
        self.log: List[Sequence[Specimen]] = []
        self.stale_generations_count = 0

    def run(self) -> None:
        self.initialize_population()

        while self.continue_condition():
            next_pop: List[Specimen] = []
            for _ in range(self.params["population_size"]):
                if self.should_crossover():
                    parent1 = self.select()
                    parent2 = self.select()
                    next_spec = self.mutation(self.crossover((parent1, parent2)))
                    next_pop.append(next_spec)
                else:
                    next_spec = self.mutation(self.select())
                    next_pop.append(next_spec)

            self.succession(next_pop)

    def continue_condition(self) -> bool:
        if self.best_specimen.fitness <= self.params["target_fitness"]:
            return False

        if self.current_generation >= self.params["max_epochs"]:
            return False

        best_got_better = self.best_specimen.fitness < self.prev_best_fitness
        self.prev_best_fitness = min(self.prev_best_fitness, self.best_specimen.fitness)

        if best_got_better:
            self.stale_generations_count = 0
        elif self.stale_generations_count + 1 >= self.params["stale_epochs_limit"]:
            return False
        else:
            self.stale_generations_count += 1

        return True

    def calc_fitness(self, spec: Specimen) -> int:
        modules = 0
        link_loads: Dict[str, float] = {link.id: 0.0 for link in self.links.values()}

        for spec_demand in spec.demands:
            demand: Demand = self.demands[spec_demand.demand_id]
            paths: Sequence[AdmissablePath] = demand.admissable_paths
            for i in range(len(spec_demand.path_uses)):
                links: Sequence[str] = paths[i].links
                use: float = spec_demand.path_uses[i]
                for link in links:
                    link_loads[link] += use * demand.demand_value

        for load in link_loads.values():
            modules += self.edge_capacity(load)

        return modules

    def edge_capacity(self, load: float) -> int:
        if self.modularity <= 0:
            raise ValueError("modularity must be positive")
        return math.ceil(load / self.modularity)

    def select(self) -> Specimen:
        tournament: List[Specimen] = []

        for _ in range(self.params["tournament_size"]):
            specimen_index = randint(0, self.params["population_size"] - 1)
            tournament.append(self.population[specimen_index])

        tournament.sort(key=lambda spec: spec.fitness)

        return tournament[0]

    def mutation(self, specimen: Specimen) -> Specimen:
        if self.should_mutate():
            return (
                self.mutation_aggregate(specimen)
                if self.aggregation
                else self.mutation_no_aggregate(specimen)
            )
        else:
            return specimen

    def get_demands_to_mutate(self, specimen: Specimen) -> Sequence[int]:
        demand_ids = list(range(0, len(specimen.demands)))
        demands_to_mutate: List[int] = []

        for _ in range(self.params["mutation_range"]):
            chosen_demand_index = randint(0, len(demand_ids) - 1)
            chosen_demand_id = demand_ids[chosen_demand_index]

            del demand_ids[chosen_demand_index]
            demands_to_mutate.append(chosen_demand_id)

        return demands_to_mutate

    def mutation_aggregate(self, specimen: Specimen) -> Specimen:
        demands_to_mutate = self.get_demands_to_mutate(specimen)

        new_specimen_demands: List[SpecimenDemand] = [d for d in specimen.demands]

        for demand_to_mutate in demands_to_mutate:
            demand_paths = len(specimen.demands[demand_to_mutate].path_uses)

            new_path = randint(0, demand_paths - 1)

            new_demand = [0.0] * demand_paths
            new_demand[new_path] = 1.0

            new_specimen_demands[demand_to_mutate] = SpecimenDemand(
                specimen.demands[demand_to_mutate].demand_id, new_demand
            )

        return Specimen(new_specimen_demands, sys.maxsize)

    def mutation_no_aggregate(self, specimen: Specimen) -> Specimen:
        demands_to_mutate = self.get_demands_to_mutate(specimen)
        demand_paths = len(specimen.demands[0].path_uses)
        new_specimen_demands: List[SpecimenDemand] = [d for d in specimen.demands]

        for demand_to_mutate_index in demands_to_mutate:
            mutation_vector = [0.0] * demand_paths
            for path_index in range(demand_paths):
                mutation_vector[path_index] = np.random.normal(
                    0, self.params["mutation_power"]
                )

            path_uses = new_specimen_demands[demand_to_mutate_index].path_uses
            mutated_path_uses = [0.0] * demand_paths
            for path_index in range(demand_paths):
                mutated_path_uses[path_index] = abs(
                    path_uses[path_index] + mutation_vector[path_index]
                )

            new_specimen_demands[demand_to_mutate_index] = SpecimenDemand(
                specimen.demands[demand_to_mutate_index].demand_id,
                Evolution.normalize_demand(mutated_path_uses),
            )

        return Specimen(new_specimen_demands, sys.maxsize)

    def crossover(self, pair: Tuple[Specimen, Specimen]) -> Specimen:
        return (
            self.crossover_aggregate(pair)
            if self.aggregation
            else self.crossover_no_aggregate(pair)
        )

    def crossover_aggregate(self, pair: Tuple[Specimen, Specimen]) -> Specimen:
        parent1, parent2 = pair

        crossover_geonome: List[SpecimenDemand] = []

        for demand_index in range(len(parent1.demands)):
            if random() > 0.5:
                crossover_geonome.append(parent1.demands[demand_index])
            else:
                crossover_geonome.append(parent2.demands[demand_index])

        return Specimen(crossover_geonome, sys.maxsize)

    def crossover_no_aggregate(self, pair: Tuple[Specimen, Specimen]) -> Specimen:
        parent1, parent2 = pair
        demands1 = parent1.demands
        demands2 = parent2.demands

        crossover_genome: List[SpecimenDemand] = []

        for demand_index in range(len(demands1)):
            crossover_demand: List[float] = []
            demand_id = demands1[demand_index].demand_id

            for path_index in range(len(parent1.demands[0].path_uses)):
                path_usage = (
                    demands1[demand_index].path_uses[path_index]
                    + demands2[demand_index].path_uses[path_index]
                ) / 2
                crossover_demand.append(path_usage)

            crossover_genome.append(SpecimenDemand(demand_id, crossover_demand))

        return Specimen(crossover_genome, sys.maxsize)

    @staticmethod
    def normalize_demand(demand: Sequence[float]) -> Sequence[float]:
        demand_sum = sum(demand)
        return [x / demand_sum for x in demand]

    def initialize_population(self):
        self.population = self.create_init_population()
        self.evaluate_population()
        self.log.append(self.population)

    def create_init_population(self) -> List[Specimen]:
        return (
            self.init_population_aggregate()
            if self.aggregation
            else self.init_population_no_aggregate()
        )

    def init_population_aggregate(self) -> List[Specimen]:
        demands = list(self.demands.values())
        paths = len(demands[0].admissable_paths)
        init_population: List[Specimen] = []

        for _ in range(self.params["population_size"]):
            new_genome: List[SpecimenDemand] = []

            for i_demand in range(len(self.demands)):
                new_gene = [0.0] * paths
                random_index = randint(0, paths - 1)
                new_gene[random_index] = 1.0
                new_genome.append(SpecimenDemand(demands[i_demand].id, new_gene))

            init_population.append(Specimen(new_genome, sys.maxsize))

        return init_population

    def init_population_no_aggregate(self) -> List[Specimen]:
        demands = list(self.demands.values())
        paths = len(demands[0].admissable_paths)
        init_population: List[Specimen] = []

        for _ in range(self.params["population_size"]):
            new_genome: List[SpecimenDemand] = []

            for i_demand in range(len(self.demands)):
                new_gene = np.random.uniform(0.0, 1.0, paths).tolist()
                new_gene = self.normalize_demand(new_gene)
                new_genome.append(SpecimenDemand(demands[i_demand].id, new_gene))

            init_population.append(Specimen(new_genome, sys.maxsize))

        return init_population

    def succession(self, next_population: List[Specimen]):
        self.population = next_population
        self.current_generation += 1
        self.evaluate_population()
        self.log.append(self.population)

    def should_crossover(self) -> bool:
        return random() < self.params["crossover_prob"]

    def should_mutate(self) -> bool:
        return random() < self.params["mutation_prob"]

    def evaluate_population(self) -> None:
        for spec_index in range(len(self.population)):
            self.population[spec_index] = Specimen(
                self.population[spec_index].demands,
                self.calc_fitness(self.population[spec_index]),
            )
            if self.population[spec_index].fitness < self.best_specimen.fitness:
                self.best_specimen = self.population[spec_index]

    def present_specimen(self, spec: Specimen) -> Sequence[LinkResult]:
        link_loads: Dict[str, float] = {link.id: 0.0 for link in self.links.values()}
        for spec_demand in spec.demands:
            demand: Demand = self.demands[spec_demand.demand_id]
            paths: Sequence[AdmissablePath] = demand.admissable_paths
            for i in range(len(spec_demand.path_uses)):
                links: Sequence[str] = paths[i].links
                use: float = spec_demand.path_uses[i]
                for link in links:
                    link_loads[link] += use * demand.demand_value
        return [
            {
                "id": self.links[link_id].id,
                "source": self.links[link_id].source,
                "target": self.links[link_id].target,
                "modules": self.edge_capacity(load),
            }
            for link_id, load in link_loads.items()
        ]

    def get_result(self) -> EvolutionResult:
        return {
            "parameters": self.params,
            "modularity": self.modularity,
            "aggregation": self.aggregation,
            "log": [[s.fitness for s in specs] for specs in self.log],
            "links": self.present_specimen(self.best_specimen),
            "modules": self.best_specimen.fitness,
        }
