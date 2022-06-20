import math
import sys
from functools import reduce
from operator import add
from typing import Dict, List

from data_model import NaiveResult, Network


class NaiveSolver:
    def __init__(self, network: Network) -> None:
        self.network = network
        self.links = {link.id: link for link in network.links}

    def solve(self) -> NaiveResult:
        demand_uses: List[int] = []
        for demand in self.network.demands:
            shortest = sys.maxsize
            shortest_index = -1
            for i in range(len(demand.admissable_paths)):
                path = demand.admissable_paths[i]
                if len(path.links) < shortest:
                    shortest = len(path.links)
                    shortest_index = i
            demand_uses.append(shortest_index)

        link_loads: Dict[str, float] = {link.id: 0.0 for link in self.network.links}
        for i in range(len(demand_uses)):
            demand_use = demand_uses[i]
            demandd = self.network.demands[i]
            path = demandd.admissable_paths[demand_use]
            for link in path.links:
                link_loads[link] += demandd.demand_value

        return {
            "modularity": self.network.modularity,
            "links": [
                {
                    "id": self.links[link_id].id,
                    "source": self.links[link_id].source,
                    "target": self.links[link_id].target,
                    "modules": self.edge_capacity(link_load),
                }
                for link_id, link_load in link_loads.items()
            ],
            "modules": reduce(
                add, (self.edge_capacity(x) for x in link_loads.values())
            ),
        }

    def edge_capacity(self, load: float) -> int:
        if self.network.modularity <= 0:
            raise ValueError("modularity must be positive")
        return math.ceil(load / self.network.modularity)
