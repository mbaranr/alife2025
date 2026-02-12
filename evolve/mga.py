# Adapted from Waldegrave et al. (2024): https://doi.org/10.1162/isal_a_00734
# Copyright (c) 2024, Riversdale Waldegrave


import os
import copy
import logging
import jsonpickle

import sqlite3
import numpy as np
from tqdm import tqdm

from multiprocessing import Lock
from evolve.fitness import ReservoirFitness

from grow.dgca import DGCA
from grow.runner import Runner
from grow.reservoir import Reservoir


class Chromosome:
    """
    Data for a "chromosome" (MLP weights and biases). 
    Implements mutation and crossover methods.
    """
    def __init__(self, 
                 weights: list[np.ndarray], 
                 biases: list[np.ndarray], 
                 mutate_rate: float, 
                 crossover_rate: float, 
                 crossover_style: str, 
                 best_fitness: np.float32 = np.nan):
        self.weights = weights  # list of weight matrices 
        self.biases = biases    # list of bias vectors 
        self.mutate_rate = mutate_rate
        self.crossover_rate = crossover_rate
        self.crossover_style = crossover_style
        self.best_fitness = best_fitness

    def mutate(self, rate: float = None) -> "Chromosome":
        """
        Mutates all weight and bias matrices in the MLP.
        """
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
        """
        Crosses over weight and bias matrices with another chromosome.
        """
        for i in range(len(self.weights)):
            if self.crossover_style == 'rows':
                row_idx = np.random.choice(
                    range(self.weights[i].shape[0]), 
                    size=int(self.weights[i].shape[0] * self.crossover_rate), 
                    replace=False
                )
                self.weights[i][row_idx, :] = other.weights[i][row_idx, :]
            elif self.crossover_style == 'cols':
                col_idx = np.random.choice(
                    range(self.weights[i].shape[1]), 
                    size=int(self.weights[i].shape[1] * self.crossover_rate), 
                    replace=False
                )
                self.weights[i][:, col_idx] = other.weights[i][:, col_idx]

        for i in range(len(self.biases)):
            idx = np.random.choice(
                range(self.biases[i].shape[0]), 
                size=int(self.biases[i].shape[0] * self.crossover_rate), 
                replace=False
            )
            self.biases[i][idx] = other.biases[i][idx]

        self.best_fitness = np.nan
        return self

    def get_new(self) -> "Chromosome":
        """
        Creates a new ChromosomeMLP with random weights and biases.
        """
        new_weights = [np.random.uniform(-1, 1, w.shape) for w in self.weights]
        new_biases = [np.random.uniform(-1, 1, b.shape) for b in self.biases]
        return Chromosome(new_weights, new_biases, self.mutate_rate, self.crossover_rate, self.crossover_style)

    def crossover(self, other: "Chromosome") -> "Chromosome":
        """
        Crosses over weight and bias matrices with another chromosome.

        Parameters:
        - other: Chromosome, the other parent Chromosome.

        Returns:
        - Chromosome: self, with crossed-over data.
        """
        for i in range(len(self.weights)):
            if self.crossover_style == 'rows':
                # row-wise crossover
                row_idx = np.random.choice(
                    range(self.weights[i].shape[0]),
                    size=int(self.weights[i].shape[0] * self.crossover_rate),
                    replace=False
                )
                # swap rows
                self.weights[i][row_idx, :] = other.weights[i][row_idx, :]
            elif self.crossover_style == 'cols':
                # column-wise crossover
                col_idx = np.random.choice(
                    range(self.weights[i].shape[1]),
                    size=int(self.weights[i].shape[1] * self.crossover_rate),
                    replace=False
                )
                # swap columns
                self.weights[i][:, col_idx] = other.weights[i][:, col_idx]

        for i in range(len(self.biases)):
            # bias crossover (element-wise)
            idx = np.random.choice(
                range(self.biases[i].shape[0]),
                size=int(self.biases[i].shape[0] * self.crossover_rate),
                replace=False
            )
            self.biases[i][idx] = other.biases[i][idx]

        self.best_fitness = np.nan
        return self

    def get_new(self) -> "Chromosome":
        """
        Creates a new Chromosome with random weights and biases.
        """
        new_weights = [np.random.uniform(-1, 1, w.shape) for w in self.weights]
        new_biases = [np.random.uniform(-1, 1, b.shape) for b in self.biases]
        return Chromosome(new_weights, new_biases, self.mutate_rate, self.crossover_rate, self.crossover_style)
   

class EvolvableDGCA(DGCA):
    def __init__(self, n_states, hidden_size=None, noise: float=0.0):
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


log_lock = Lock() 

class ChromosomalMGA:
    
    def __init__(self, 
                 popsize: int,
                 model: DGCA,
                 seed_graph: Reservoir,
                 runner: Runner,
                 fitness_fn: ReservoirFitness,
                 mutate_rate: float, 
                 cross_rate: float, 
                 cross_style: str,
                 run_id: int,
                 db_file: str = "fitness.db",
                 n_trials: int = 2000,
                 heavy_log: bool = False,
                 bsz: int = 20):
        self.popsize = popsize
        self.model = model
        self.seed_graph = seed_graph
        self.runner = runner
        self.fitness_fn = fitness_fn
        self.run_id = run_id
        self.trial = 0
        self.n_trials = n_trials
        self.heavy_log = heavy_log
        
        # logging
        os.makedirs("logs", exist_ok=True)
        self.log_file = os.path.join("logs", f"{self.run_id}.stdout")        
        self.logger = logging.getLogger(f"run_{self.run_id}")  
        self.logger.setLevel(logging.INFO)
        
        # prevents duplicate handlers
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        file_handler = logging.FileHandler(self.log_file, mode='w') 
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(file_handler)
        
        self.db_file = db_file

        self.fitness_cache = []  
        self.bsz = bsz  # batch size for logging

        self.best = {}

        # nan tolerant fitness comparison
        if self.fitness_fn.high_good:
            self.better = lambda a, b: np.isnan(b) or a >= b
            self.best['fitness'] = -float('inf')
        else:
            self.better = lambda a, b: np.isnan(b) or a <= b
            self.best['fitness'] = float('inf')

        self.base_chromosomes = self.model.get_chromosomes(mutate_rate, cross_rate, cross_style)
        self.pop_chromosomes = np.array([[bc.get_new() for _ in range(self.popsize)] for bc in self.base_chromosomes]).T
        self.num_chromosomes = len(self.base_chromosomes)

        self._initialize_database()

        print(f"Results will be stored in SQLite: {self.db_file}")

    def _initialize_database(self):
        with log_lock, sqlite3.connect(self.db_file, timeout=10) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fitness (
                    run_id INTEGER,
                    epoch INTEGER,
                    fitness REAL,
                    best_fitness REAL,
                    kr REAL,
                    gm REAL,
                    size INT,
                    skip_count INTEGER
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    run_id INTEGER,
                    epoch INTEGER,
                    model TEXT,
                    reservoir TEXT
                )
            """)
            
            conn.commit()

    from multiprocessing import Lock

    def log_fitness(self, fitness: float, reservoir: Reservoir):
        """
        Efficiently logs fitness results to a log file and SQLite.
        Uses buffered logging and batch writes to reduce latency.
        """
        # no mutex, adds too much overhead
        # kr, gm = get_metrics(reservoir)

        with log_lock:  
            self.logger.info(f"Epoch: {self.trial}, Fitness: {fitness}, Best Fitness: {self.best['fitness']}")
            data = (
                self.run_id, self.trial, fitness, 
                self.best['fitness'], 0., 0., reservoir.size(),
                self.fitness_fn.skip_count
            )
            self.fitness_cache.append(data)

            # write in batches
            if len(self.fitness_cache) >= self.bsz:
                with sqlite3.connect(self.db_file, timeout=10) as conn:
                    cursor = conn.cursor()
                    cursor.executemany("""
                        INSERT INTO fitness (run_id, epoch, fitness, best_fitness, kr, gm, size, skip_count)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, self.fitness_cache)
                    conn.commit()
                    self.fitness_cache.clear()  # reset cache
    
    def log_model(self, model: DGCA, reservoir: Reservoir, trial=None):
        """
        Save the final model and reservoir.
        """
        with log_lock, sqlite3.connect(self.db_file, timeout=10) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO models (run_id, epoch, model, reservoir) 
                VALUES (?, ?, ?, ?)
            """, (self.run_id, self.trial if not trial else trial, jsonpickle.encode(model), jsonpickle.encode(reservoir)))

            conn.commit()

        print(f"Model and reservoir stored in SQLite: {self.db_file}")

    def run(self, progress: bool=False):
        """
        Main evolution loop.
        """
        pbar = tqdm(range(self.n_trials), postfix={'fit':0, 'best':0}) if progress else range(self.n_trials)
        for _ in pbar:
            f = self.contest()
            self.trial += 1
            if progress:
                pbar.set_postfix({'fit': f, 'best': self.best["fitness"]})
        self.log_model(self.best['model'], self.best['reservoir'], trial=-1)

    def contest(self) -> float:
        """
        Runs a single contest between two individuals created out of randomly selected chromosomes
        Returns the fitness of the fitter one.
        """
        # select two sets of chromosomes at random
        idx = np.random.randint(low=0,high=self.popsize,size=(2,self.num_chromosomes))
        contestant_chromosomes = np.take_along_axis(self.pop_chromosomes, idx, axis=0)
        fitness = (np.nan, np.nan)
        
        # keep looping until they are not nan, doesn't count as new 'runs'
        while np.all(np.isnan(fitness)):
            fitness = self.run_individual(contestant_chromosomes[0]), self.run_individual(contestant_chromosomes[1])
            # if both contestants' fitness are nan, mutate the chromosomes
            if np.all(np.isnan(fitness)):
                for i, chr in enumerate(contestant_chromosomes.flat):
                    if np.isnan(fitness[i // self.num_chromosomes]):
                        if np.isnan(chr.best_fitness):
                            chr.mutate(rate=1.0)  # do 100% mutation in this case
                        else:
                            chr.mutate()  # default mutation rate
        
        win, lose = (0,1) if self.better(*fitness) else (1,0)
        for c in range(self.num_chromosomes):
            if idx[win,c]==idx[lose,c]:
                # chromosomes were the same, don't change anything
                continue
            else:
                chr_win, chr_lose = contestant_chromosomes[win,c], contestant_chromosomes[lose,c]
                # don't change the losing individual if it has previously been part of an 
                # individual with higher fitness than the winner
                if self.better(fitness[win], chr_lose.best_fitness):
                    # call crossover & mutate on the loser (this changes it in place)
                    chr_lose.crossover(chr_win).mutate()
        return fitness[win]

    def run_individual(self, chromosomes: list[Chromosome]) -> float:
        """
        Returns the fitness of one set of chromosomes.
        """
        self.model.set_chromosomes(*chromosomes)
        self.runner.reset()
        final_res = self.runner.run(self.model, self.seed_graph)
        fitness = self.fitness_fn(final_res)
        # update chromosomes' best_fitness score
        for chr in chromosomes:
            if self.better(fitness, chr.best_fitness):
                chr.best_fitness = fitness
        if not(np.isnan(fitness)) and (self.db_file is not None):
            # updating best fitness attributes
            if self.better(fitness, self.best['fitness']):
                self.best['fitness'] = fitness
                self.best['model'] = copy.deepcopy(self.model)
                self.best['reservoir'] = final_res.copy()

            self.log_fitness(fitness, final_res)
            if self.heavy_log:
                self.log_model(self.model, final_res)
        return fitness