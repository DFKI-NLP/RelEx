from typing import List

import os

# import threading
import argparse
from multiprocessing import Manager
from joblib import Parallel, delayed
from probing_task_evaluation import run_evaluation


def _get_parser():
    parser = argparse.ArgumentParser(description="Run evaluation on probing tasks")

    parser.add_argument(
        "--experiment-dir",
        type=str,
        required=True,
        help="directory containing the model archive file",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="directory containing the probing task data files",
    )
    parser.add_argument(
        "--cuda-devices",
        nargs="+",
        required=True,
        help="a list of cuda device to load the models on",
    )
    parser.add_argument(
        "--predictor",
        type=str,
        default="relation_classifier",
        help="predictor to use for probing tasks",
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="batch size to use for predictions"
    )
    parser.add_argument("--prototyping", action="store_true")

    parser.add_argument("--cache-representations", action="store_true")

    return parser


def runner(
    model_dir: str,
    data_dir: str,
    output_dir: str,
    predictor: str,
    batch_size: int,
    prototyping: bool,
    cache_representations: bool,
    q: Queue,
):
    cuda_device = q.get()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

    run_evaluation(
        model_dir=model_dir,
        data_dir=data_dir,
        output_dir=output_dir,
        predictor=predictor,
        batch_size=batch_size,
        cuda_device=cuda_device,
        prototyping=prototyping,
        cache_representations=cache_representations,
    )

    q.put(cuda_device)


def run_evaluation_parallel(
    experiment_dir: str,
    data_dir: str,
    cuda_devices: List[int],
    result_filename: str = "result.json",
    trial_dirname: str = "trial",
    predictor: str = "relation_classifier",
    batch_size: int = 128,
    prototyping: bool = False,
    cache_representations: bool = True,
):
    # code taken from
    # https://gist.github.com/DmitryUlyanov/a5c37f08dcf0e242a50bf390c176daae#file-run_batch2-py

    # Fix print
    # _print = print
    # _rlock = threading.RLock()

    # def print(*args, **kwargs):
    #     with _rlock:
    #         _print(*args, **kwargs)

    trial_dirs = []
    for dirpath, _, filenames in os.walk(experiment_dir):
        for filename in filenames:
            if filename != result_filename:
                continue

            trial_dir = os.path.join(dirpath, trial_dirname)
            trial_dirs.append(trial_dir)

    output_dirs = [
        os.path.abspath(os.path.join(trial_dir, os.pardir)) for trial_dir in trial_dirs
    ]

    # Put indices in queue
    n_gpus = len(cuda_devices)
    manager = Manager()
    q = manager.Queue(maxsize=n_gpus)
    for i in range(n_gpus):
        q.put(i)

    Parallel(n_jobs=n_gpus, backend="multiprocessing")(
        delayed(runner)(
            model_dir=trial_dir,
            data_dir=data_dir,
            output_dir=output_dir,
            predictor=predictor,
            batch_size=batch_size,
            prototyping=prototyping,
            cache_representations=cache_representations,
            q=q,
        )
        for trial_dir, output_dir in zip(trial_dirs, output_dirs)
    )


if __name__ == "__main__":
    parser = _get_parser()
    args = parser.parse_args()
    run_evaluation_parallel(
        experiment_dir=args.experiment_dir,
        data_dir=args.data_dir,
        cuda_devices=args.cuda_devices,
        predictor=args.predictor,
        batch_size=args.batch_size,
        prototyping=args.prototyping,
        cache_representations=args.cache_representations,
    )
