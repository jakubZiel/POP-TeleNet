from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypedDict


@dataclass(frozen=True)
class AdmissablePath:
    links: Sequence[str]


@dataclass(frozen=True)
class Demand:
    id: str
    demand_value: float
    admissable_paths: Sequence[AdmissablePath]


@dataclass(frozen=True)
class Link:
    id: str
    source: str
    target: str


@dataclass(frozen=True)
class Network:
    demands: Sequence[Demand]
    links: Sequence[Link]
    modularity: int
    aggregation: bool


@dataclass(frozen=True)
class SpecimenDemand:
    demand_id: str
    path_uses: Sequence[float]


@dataclass(frozen=True)
class Specimen:
    demands: Sequence[SpecimenDemand]
    fitness: int


class AlgorithmParameters(TypedDict):
    population_size: int
    crossover_prob: float
    tournament_size: int
    mutation_prob: float
    mutation_power: float
    mutation_range: int
    target_fitness: float
    max_epochs: int
    stale_epochs_limit: int


class LinkResult(TypedDict):
    id: str
    source: str
    target: str
    modules: int


class EvolutionResult(TypedDict):
    parameters: AlgorithmParameters
    modularity: int
    aggregation: bool
    log: Sequence[Sequence[float]]
    links: Sequence[LinkResult]
    modules: int


class NaiveResult(TypedDict):
    modularity: int
    links: Sequence[LinkResult]
    modules: int
