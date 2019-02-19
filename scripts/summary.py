from typing import Dict, Any

import os
import argparse
import json
import logging
import numpy as np
from functools import partial
from relex.evaluation import semeval2010_task8_evaluation
from relex.evaluation import tacred_evaluation

logger = logging.getLogger(__name__)


def _get_parser():
    parser = argparse.ArgumentParser(description="Experiment summary")

    parser.add_argument(
        "--experiment-dir",
        type=str,
        required=True,
        help="directory containing the runs for an experiment",
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="dataset to be evaluated"
    )
    parser.add_argument(
        "--test-file",
        type=str,
        required=True,
        help="file containing the examples for evaluation",
    )
    parser.add_argument(
        "--official-eval-script",
        type=str,
        default=None,
        help="path to the official evaluation script, if required by the dataset",
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="batch size to use for predictions"
    )
    parser.add_argument(
        "--cuda-device", type=int, default=-1, help="a cuda device to load the model on"
    )

    return parser


def experiment_summary(
    experiment_dir: str,
    scorer,
    result_filename: str = "result.json",
    trial_dirname: str = "trial",
    metrics_filename: str = "metrics.json",
) -> Dict[str, Any]:
    def name_from_path(path: str) -> str:
        return os.path.basename(os.path.normpath(path))

    def summary_stats(x: np.array) -> Dict[str, Any]:
        return {"mean": np.mean(x), "stddev": np.std(x)}

    precision = []
    recall = []
    f1 = []

    trials = []

    for dirpath, _, filenames in os.walk(experiment_dir):
        for filename in filenames:
            if filename != result_filename:
                continue

            # result_file = os.path.join(dirpath, result_filename)
            trial_dir = os.path.join(dirpath, trial_dirname)
            # metrics_file = os.path.join(trial_dir, metrics_filename)

            trial_precision, trial_recall, trial_f1 = scorer(trial_dir)
            precision.append(trial_precision)
            recall.append(trial_recall)
            f1.append(trial_f1)

            trials.append(
                {
                    "trial": name_from_path(dirpath),
                    "precision": trial_precision,
                    "recall": trial_recall,
                    "f1": trial_f1,
                }
            )

    return {
        "experiment": name_from_path(experiment_dir),
        "trials": trials,
        "stats": {
            "precision": summary_stats(precision),
            "recall": summary_stats(recall),
            "f1": summary_stats(f1),
        },
    }


def main():
    parser = _get_parser()
    args = parser.parse_args()

    if args.dataset == "semeval2010":
        scorer = partial(
            semeval2010_task8_evaluation.evaluate,
            test_file=args.test_file,
            eval_script_file=args.official_eval_script,
            batch_size=args.batch_size,
            cuda_device=args.cuda_device,
        )

    elif args.dataset == "tacred":
        scorer = partial(
            tacred_evaluation.evaluate,
            test_file=args.test_file,
            batch_size=args.batch_size,
            cuda_device=args.cuda_device,
        )

    summary = experiment_summary(args.experiment_dir, scorer)

    logger.info(f"Experiment Summary:\n{json.dumps(summary, indent=4, sort_keys=True)}")

    with open(os.path.join(args.experiment_dir, "summary.json"), "w") as summary_f:
        json.dump(summary, summary_f)


if __name__ == "__main__":
    main()
