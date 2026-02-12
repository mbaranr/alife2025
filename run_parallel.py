from joblib import Parallel, delayed
from types import SimpleNamespace

from run import run_ga

if __name__ == "__main__":

    total_runs = 150

    args_dict = {
        "pop_size": 10,
        "mutate_rate": 0.02,
        "cross_rate": 0.5,
        "cross_style": "cols",
        "n_trials": 1000,
        "input_nodes": 0,
        "output_nodes": 0, 
        "noise": 0.05,
        "order": 10,
        "task": "narma",
        "max_size": 200,
        "metric": None, 
        "n_states": 3,
        "output_file": "fitness.db",
        "num_jobs": total_runs,
        "heavy_log": False
    }

    args = SimpleNamespace(**args_dict)

    num_parallel_jobs = 8  # match with cpu cores

    Parallel(n_jobs=num_parallel_jobs)(
        delayed(run_ga)(run_id, args) for run_id in range(total_runs)
    )

    print("All GA runs completed.")