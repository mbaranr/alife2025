# Adapted from Waldegrave et al. (2024): https://doi.org/10.1162/isal_a_00734
# Copyright (c) 2024, Riversdale Waldegrave

import os
import copy
import jsonpickle
import numpy as np
from tqdm import tqdm

from evolve.fitness import ReservoirFitness
from grow.dgca import DGCA
from grow.runner import Runner
from grow.reservoir import Reservoir


class Chromosome:
    """
    Data for a "chromosome" (MLP weights and biases).
    Implements mutation and crossover methods.
    """
    def __init__(
        self,
        weights: list[np.ndarray],
        biases: list[np.ndarray],
        mutate_rate: float,
        crossover_rate: float,
        crossover_style: str,
        best_fitness: np.float32 = np.nan,
    ):
        self.weights = weights
        self.biases = biases
        self.mutate_rate = mutate_rate
        self.crossover_rate = crossover_rate
        self.crossover_style = crossover_style
        self.best_fitness = best_fitness

    def mutate(self, rate: float = None) -> "Chromosome":
        if rate is None:
            rate = self.mutate_rate

        for i, w in enumerate(self.weights):
            mask = np.random.choice([True, False], size=w.shape, p=[rate, 1 - rate])
            random_vals = np.random.uniform(-1, 1, size=w.shape)
            self.weights[i][mask] = random_vals[mask]

        for i, b in enumerate(self.biases):
            mask = np.random.choice([True, False], size=b.shape, p=[rate, 1 - rate])
            random_vals = np.random.uniform(-1, 1, size=b.shape)
            self.biases[i][mask] = random_vals[mask]

        self.best_fitness = np.nan
        return self

    def crossover(self, other: "Chromosome") -> "Chromosome":
        for i in range(len(self.weights)):
            if self.crossover_style == "rows":
                row_idx = np.random.choice(
                    range(self.weights[i].shape[0]),
                    size=int(self.weights[i].shape[0] * self.crossover_rate),
                    replace=False,
                )
                self.weights[i][row_idx, :] = other.weights[i][row_idx, :]
            elif self.crossover_style == "cols":
                col_idx = np.random.choice(
                    range(self.weights[i].shape[1]),
                    size=int(self.weights[i].shape[1] * self.crossover_rate),
                    replace=False,
                )
                self.weights[i][:, col_idx] = other.weights[i][:, col_idx]

        for i in range(len(self.biases)):
            idx = np.random.choice(
                range(self.biases[i].shape[0]),
                size=int(self.biases[i].shape[0] * self.crossover_rate),
                replace=False,
            )
            self.biases[i][idx] = other.biases[i][idx]

        self.best_fitness = np.nan
        return self

    def get_new(self) -> "Chromosome":
        new_weights = [np.random.uniform(-1, 1, w.shape) for w in self.weights]
        new_biases = [np.random.uniform(-1, 1, b.shape) for b in self.biases]
        return Chromosome(new_weights, new_biases, self.mutate_rate, self.crossover_rate, self.crossover_style)


class EvolvableDGCA(DGCA):
    def __init__(self, n_states, hidden_size=None, noise: float = 0.0):
        super().__init__(n_states=n_states, hidden_size=hidden_size, noise=noise)

    def set_chromosomes(self, chr_action: Chromosome, chr_state: Chromosome):
        self.action_mlp.set_parameters(chr_action.weights, chr_action.biases)
        self.state_mlp.set_parameters(chr_state.weights, chr_state.biases)

    def get_chromosomes(self, mutate_rate: float, cross_rate: float, cross_style: str):
        weights_action, biases_action = self.action_mlp.get_parameters()
        weights_state, biases_state = self.state_mlp.get_parameters()
        chr_action = Chromosome(weights_action, biases_action, mutate_rate, cross_rate, cross_style)
        chr_state = Chromosome(weights_state, biases_state, mutate_rate, cross_rate, cross_style)
        return chr_action, chr_state


class ChromosomalMGA:
    def __init__(
        self,
        popsize: int,
        model: DGCA,
        seed_graph: Reservoir,
        runner: Runner,
        fitness_fn: ReservoirFitness,
        mutate_rate: float,
        cross_rate: float,
        cross_style: str,
        run_id: int,
        output_dir: str = "results",
        n_trials: int = 2000,
        heavy_log: bool = False,
        n_reps_if_noisy: int = 5,
    ):
        self.popsize = popsize
        self.model = model
        self.seed_graph = seed_graph
        self.runner = runner
        self.fitness_fn = fitness_fn
        self.run_id = run_id
        self.trial = 0
        self.n_trials = n_trials
        self.heavy_log = heavy_log

        # persist once per run
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.results_file = os.path.join(self.output_dir, f"best_run_{self.run_id}.json")

        noise = getattr(self.model, "noise", 0.0)
        self.n_reps = int(n_reps_if_noisy) if (noise is not None and noise > 0.0) else 1

        # best tracking (contest uses avg fitness if noisy)
        self.best = {}
        if self.fitness_fn.high_good:
            self.better = lambda a, b: np.isnan(b) or a >= b
            self.best["avg_fitness"] = -float("inf")
        else:
            self.better = lambda a, b: np.isnan(b) or a <= b
            self.best["avg_fitness"] = float("inf")

        self.best["fitnesses"] = []   # len 1 if not noisy else len n_reps
        self.best["reservoirs"] = []  # len 1 if not noisy else len n_reps
        self.best["model"] = None

        self.base_chromosomes = self.model.get_chromosomes(mutate_rate, cross_rate, cross_style)
        self.pop_chromosomes = np.array(
            [[bc.get_new() for _ in range(self.popsize)] for bc in self.base_chromosomes]
        ).T
        self.num_chromosomes = len(self.base_chromosomes)

        print(f"[run {self.run_id}] Best-of-run results will be stored in: {self.results_file}")
        if self.n_reps > 1:
            print(f"[run {self.run_id}] Noisy model (noise={noise}); using {self.n_reps} reps and contesting on avg fitness.")

    def save_best(self, trial: int = -1):
        payload = {
            "run_id": int(self.run_id),
            "trial": int(trial),
            "n_reps": int(self.n_reps),
            "best_avg_fitness": float(self.best.get("avg_fitness", np.nan)),
            "best_fitnesses": [float(x) for x in (self.best.get("fitnesses") or [])],
            "best_model": jsonpickle.encode(self.best.get("model")),
            "best_reservoirs": [jsonpickle.encode(r) for r in (self.best.get("reservoirs") or [])],
        }

        tmp_path = self.results_file + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(jsonpickle.encode(payload, indent=2))
        os.replace(tmp_path, self.results_file)

        print(f"[run {self.run_id}] Saved best: avg_fitness={payload['best_avg_fitness']} to {self.results_file}")

    def run(self, progress: bool = False):
        pbar = tqdm(range(self.n_trials), postfix={"fit": 0, "best": 0}) if progress else range(self.n_trials)
        for _ in pbar:
            f = self.contest()
            self.trial += 1
            if progress:
                pbar.set_postfix({"fit": f, "best": self.best["avg_fitness"]})

        # persist once at the end
        self.save_best(trial=-1)

    def contest(self) -> float:
        """
        Contest two individuals (built from randomly selected chromosomes).
        Returns the winner's average fitness (or single fitness if not noisy).
        """
        idx = np.random.randint(low=0, high=self.popsize, size=(2, self.num_chromosomes))
        contestant_chromosomes = np.take_along_axis(self.pop_chromosomes, idx, axis=0)

        fitness = (np.nan, np.nan)

        # keep looping until not both NaN
        while np.all(np.isnan(fitness)):
            fitness = (
                self.run_individual(contestant_chromosomes[0]),
                self.run_individual(contestant_chromosomes[1]),
            )

            if np.all(np.isnan(fitness)):
                for i, chr in enumerate(contestant_chromosomes.flat):
                    if np.isnan(fitness[i // self.num_chromosomes]):
                        if np.isnan(chr.best_fitness):
                            chr.mutate(rate=1.0)
                        else:
                            chr.mutate()

        win, lose = (0, 1) if self.better(*fitness) else (1, 0)

        for c in range(self.num_chromosomes):
            if idx[win, c] == idx[lose, c]:
                continue

            chr_win, chr_lose = contestant_chromosomes[win, c], contestant_chromosomes[lose, c]

            # don't change loser if it has previously been part of an individual with higher fitness than winner
            if self.better(fitness[win], chr_lose.best_fitness):
                chr_lose.crossover(chr_win).mutate()

        return float(fitness[win])

    def run_individual(self, chromosomes: list[Chromosome]) -> float:
        """
        Evaluate a set of chromosomes.
        If model.noise > 0, perform self.n_reps evaluations and return the average fitness.
        If a new best-of-run is found, store ALL per-rep reservoirs and fitnesses (len 1 or len n_reps).
        """
        self.model.set_chromosomes(*chromosomes)

        fitnesses: list[float] = []
        reservoirs: list[Reservoir] = []

        for _ in range(self.n_reps):
            self.runner.reset()
            final_res = self.runner.run(self.model, self.seed_graph)
            f = float(self.fitness_fn(final_res))
            # if one of the reps returns NaN, treat the whole individual's fitness as NaN
            if np.isnan(f):
                return np.nan
            
            fitnesses.append(f)
            reservoirs.append(final_res)

        avg_fitness = float(np.mean(fitnesses))

        # chromosome cache based on avg_fitness
        for chr in chromosomes:
            if self.better(avg_fitness, chr.best_fitness):
                chr.best_fitness = avg_fitness

        # update global best-of-run based on avg_fitness; store all per-rep artifacts
        if self.better(avg_fitness, self.best["avg_fitness"]):
            self.best["avg_fitness"] = avg_fitness
            self.best["fitnesses"] = [float(x) for x in fitnesses]
            self.best["reservoirs"] = [r.copy() for r in reservoirs]
            self.best["model"] = copy.deepcopy(self.model)

        if self.heavy_log:
            print(
                f"[run {self.run_id}] epoch={self.trial} avg_fit={avg_fitness} "
                f"fits={fitnesses} best_avg={self.best['avg_fitness']}"
            )
        else:
            print(f"[run {self.run_id}] epoch={self.trial} avg_fit={avg_fitness} best_avg={self.best['avg_fitness']}")

        return avg_fitness