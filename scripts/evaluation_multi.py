from typing import Dict, Any

import os
import argparse
import logging
import json
from functools import partial
from relex.evaluation import semeval2010_task8_evaluation
from relex.evaluation import tacred_evaluation

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
logger.addHandler(ch)


def _get_parser():
    parser = argparse.ArgumentParser(description="Run evaluation")

    parser.add_argument(
        "--experiment-dir",
        type=str,
        required=True,
        help="directory containing experiment results",
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


def evaluate_multi(
    experiment_dir: str,
    scorer,
    result_filename: str = "result.json",
    trial_dirname: str = "trial",
    metrics_filename: str = "metrics.json",
) -> Dict[str, Any]:

    summary = []
    for dirpath, _, filenames in os.walk(experiment_dir):
        for filename in filenames:
            if filename != result_filename:
                continue

            logger.info(f"Found experiment in {dirpath}")

            trial_dir = os.path.join(dirpath, trial_dirname)

            precision, recall, f1 = scorer(trial_dir)
            eval_results = dict(precision=precision, recall=recall, f1=f1)
            summary.append(dict(experiment=dirpath, results=eval_results))

            eval_result_path = os.path.join(dirpath, "evaluation_result.json")
            logger.info(f"Writing evaluation result to {eval_result_path}")

            with open(eval_result_path, "w") as result_f:
                json.dump(eval_results, result_f, indent=4, sort_keys=True)

    return summary


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

    summary = evaluate_multi(args.experiment_dir, scorer)

    logger.info(f"Evaluation Summary: {json.dumps(summary, indent=4, sort_keys=True)}")


if __name__ == "__main__":
    main()
