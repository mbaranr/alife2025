import numpy as np
from grow.reservoir import Reservoir, check_conditions
from measure.tasks import *
from measure.metrics import *


NRMSE = lambda y, y_fit: np.sqrt(np.mean((y - y_fit) ** 2) / np.var(y))


class ReservoirFitness:
    def __init__(self, high_good: bool, self_loops: bool = True, conditions: dict = None, verbose: bool = False):
        self.high_good = high_good
        self.self_loops = self_loops
        self.conditions = conditions or {}
        self.verbose = verbose
        self.skip_count = 0
        self.memo = {'fitness': [], 'graph': [], 'model': []}

    def _prepare_reservoir(self, res: Reservoir) -> Reservoir:
        if not check_conditions(res, self.conditions, self.verbose):
            self.skip_count += 1
            return None
        res_ = res.no_selfloops() if not self.self_loops else res.copy()
        return res_

    def __call__(self, res: Reservoir) -> float:
        raise NotImplementedError


class TaskFitness(ReservoirFitness):
    """
    Fitness based on performance in a benchmark task.
    """

    def __init__(self,
                 series: callable,
                 order: int = None,
                 measurements: int = 5,
                 fixed_series: bool = True,
                 **kwargs):
        super().__init__(high_good=False, **kwargs)  # NRMSE: lower is better
        self.series = series
        self.order = order
        self.measurements = measurements
        self.fixed_series = fixed_series
        self.input, self.target = self.series(order=self.order)

    def _generate_series(self):
        return self.series(order=self.order) if not self.fixed_series else (self.input, self.target)

    def __call__(self, res: Reservoir) -> float:
        res_ = self._prepare_reservoir(res)
        if res_ is None:
            return np.nan

        errors = []
        for _ in range(self.measurements):
            self.input, self.target = self._generate_series()
            res_.reset()
            predictions = res_.bipolar().train(self.input, target=self.target)
            err = np.nan if predictions is None else min(NRMSE(self.target[:, res.washout:], predictions), 1)
            errors.append(err)

        valid_errors = [e for e in errors if not np.isnan(e)]
        final_err = np.nanmean(valid_errors) if valid_errors else np.nan

        if self.verbose:
            print(f'Skipped {self.skip_count}')

        self.skip_count = 0
        return final_err


class MetricFitness(ReservoirFitness):
    """
    Fitness based on intrinsic reservoir metrics (KR, GR, etc.).
    """

    METRIC_FUNCS = {
        "kr": lambda res: kernel_rank(res) / res.size(),
        "gr": lambda res: generalization_rank(res) / res.size(),
        "lmc": lambda res: linear_memory_capacity(res, normalize=True),
        "sr": lambda res: 1 - np.abs(1 - spectral_radius(res)),
        "qmc": lambda res: quadratic_memory_capacity(res, normalize=True),
        "xmc": lambda res: cross_memory_capacity(res, normalize=True),
    }

    def __init__(self, metric: str, **kwargs):
        super().__init__(high_good=True, **kwargs)
        self.metric = metric

    def _compute_metric(self, res: Reservoir) -> float:
        if self.metric == "all":
            return sum(func(res) for func in self.METRIC_FUNCS.values())
        
        if self.metric == "nlmc":
            return cross_memory_capacity(res, normalize=True) + \
                   quadratic_memory_capacity(res, normalize=True) 

        func = self.METRIC_FUNCS.get(self.metric)
        if func is None:
            raise ValueError(f"Unknown metric: {self.metric}")

        return func(res.bipolar())

    def __call__(self, res: Reservoir) -> float:
        res_ = self._prepare_reservoir(res)
        if res_ is None:
            return np.nan

        result = self._compute_metric(res_)
        if self.verbose:
            print(f'Skipped {self.skip_count}')
        self.skip_count = 0
        return result