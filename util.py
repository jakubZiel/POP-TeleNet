import math
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from data_model import AlgorithmParameters, Network
from evolution import Evolution
from parsing import NetworkParser


class myThread(threading.Thread):
    def __init__(self, param_space: List[Tuple[int, float, float, float]], id: str):
        threading.Thread.__init__(self)
        self.param_space = param_space
        self.best_value = math.inf
        self.id = id
        self.counter = 1
        self.best_params = (0, 0.0, 0.0, 0.0)

    def run(self):
        parser = NetworkParser(Path("polska/polska.xml"))

        net = Network(
            demands=parser.demands(),
            links=parser.links(),
            modularity=10,
            aggregation=False,
        )

        with open("./results/thread" + self.id, "w") as file:
            for param_combination in self.param_space:
                params = AlgorithmParameters(
                    population_size=param_combination[0],
                    crossover_prob=param_combination[1],
                    tournament_size=2,
                    mutation_prob=param_combination[2],
                    mutation_power=param_combination[3],
                    mutation_range=1,
                    target_fitness=0,
                    max_epochs=10000,
                    stale_epochs_limit=1000,
                )

                evo = Evolution(network=net, parameters=params)

                evo.run()
                result = evo.get_result()["modules"]

                if evo.get_result()["modules"] < self.best_value:
                    self.best_params = param_combination
                    self.best_value = result

                file.write(
                    "thread"
                    + self.id
                    + ":: "
                    + str(datetime.now().time())
                    + " val: "
                    + str(result)
                    + " params: "
                    + str(param_combination)
                )

                self.counter += 1

    def get_best(self) -> float:
        return self.best_value
