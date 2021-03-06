import argparse
from relex.evaluation import semeval2010_task8_evaluation
from relex.evaluation import tacred_evaluation


def _get_parser():
    parser = argparse.ArgumentParser(description="Run evaluation")

    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="directory containing the model archive file",
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


def main():
    parser = _get_parser()
    args = parser.parse_args()

    if args.dataset == "semeval2010":
        print(
            semeval2010_task8_evaluation.evaluate(
                model_dir=args.model_dir,
                test_file=args.test_file,
                eval_script_file=args.official_eval_script,
                batch_size=args.batch_size,
                cuda_device=args.cuda_device,
            )
        )
    elif args.dataset == "tacred":
        tacred_evaluation.evaluate(
            model_dir=args.model_dir,
            test_file=args.test_file,
            batch_size=args.batch_size,
            cuda_device=args.cuda_device,
        )


if __name__ == "__main__":
    main()
