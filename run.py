import argparse
from types import SimpleNamespace

from measure.tasks import narmax, santa_fe

from grow.runner import Runner
from grow.reservoir import get_seed

from evolve.fitness import TaskFitness, MetricFitness
from evolve.mga import ChromosomalMGA, EvolvableDGCA


def run_ga(run_id, args):
    """
    Runs a single GA experiment instance.
    """
    print(f"Starting GA run {run_id}...")

    conditions = {'max_size': args.max_size, 
                  'min_size': args.input_nodes+args.output_nodes+ (10 if not args.order else args.order), 
                #   "io_path": True
                  }

    if args.task:
        fitness_fn = TaskFitness(series=narmax if args.task == "narma" else santa_fe,
                                 conditions=conditions, 
                                 verbose=False,
                                 order=args.order,
                                 fixed_series=True)
    elif args.metric:
        fitness_fn = MetricFitness(metric=args.metric,
                                   conditions=conditions, 
                                   verbose=False)

    reservoir = get_seed(args.input_nodes, args.output_nodes, args.n_states)
    model = EvolvableDGCA(n_states=reservoir.n_states, hidden_size=64, noise=args.noise)
    runner = Runner(max_steps=100, max_size=300)

    mga = ChromosomalMGA(popsize=args.pop_size,
                        seed_graph=reservoir,
                        model=model,
                        runner=runner,
                        fitness_fn=fitness_fn,
                        mutate_rate=args.mutate_rate,
                        cross_rate=args.cross_rate,
                        run_id=run_id,
                        n_trials=args.n_trials,
                        cross_style=args.cross_style,
                        db_file=args.output_file,
                        heavy_log=args.heavy_log)
    
    mga.run()
    print(f"Completed GA run {run_id}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int, required=True, help="ID of the GA run")
    args = parser.parse_args()

    args_dict = {
        "pop_size": 10,
        "mutate_rate": 0.02,
        "cross_rate": 0.5,
        "cross_style": "cols",
        "n_trials": 1000,
        "input_nodes": 0,
        "output_nodes": 0,
        "noise": .0,
        "order": None,
        "task": None,
        "max_size": 200,
        "metric": "all", 
        "n_states": 3,
        "output_file": "fitness.db",
        "heavy_log": False
    }

    args = SimpleNamespace(**args_dict, run_id=args.run_id)

    run_ga(args.run_id, args)