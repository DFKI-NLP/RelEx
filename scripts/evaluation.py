import argparse
from relex.evaluation.semeval2010_task8_evaluation import evaluate


def _get_parser():
    parser = argparse.ArgumentParser(description="Run tuna")

    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="directory containing the model archive file",
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

    print(
        evaluate(
            model_dir=args.model_dir,
            test_file=args.test_file,
            eval_script_file=args.official_eval_script,
            batch_size=args.batch_size,
            cuda_device=args.cuda_device,
        )
    )


if __name__ == "__main__":
    main()
