import os
import logging
import argparse
import json
import numpy as np
import relex
import reval
from relex.predictors.utils import load_predictor

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


ALL_PROBING_TASKS = [
    "Length",
    "EntityDistance",
    "ArgumentOrder",
    "EntityExistsBetweenHeadTail",
    "EntityCountORGBetweenHeadTail",
    "EntityCountPERBetweenHeadTail",
    "EntityCountDATEBetweenHeadTail",
    "EntityCountMISCBetweenHeadTail",
    "EntityCountLOCBetweenHeadTail",
    "PosTagHeadLeft",
    "PosTagHeadRight",
    "PosTagTailLeft",
    "PosTagTailRight",
    "ArgTypeHead",
    "ArgTypeTail",
    "TreeDepth",
    "SDPTreeDepth",
]


def _get_parser():
    parser = argparse.ArgumentParser(description="Run evaluation on probing tasks")

    parser.add_argument(
        "--model-dir",
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
        "--output-dir",
        type=str,
        default=None,
        help="directory to use for storing the probing task results",
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
    parser.add_argument(
        "--cuda-device", type=int, default=-1, help="a cuda device to load the model on"
    )
    parser.add_argument("--prototyping", action="store_true")

    return parser


def main():
    parser = _get_parser()
    args = parser.parse_args()

    def prepare(params, samples):
        pass

    def batcher(params, batch, heads, tails, ner, pos, dep, dep_head):
        inputs = []
        for sent, head, tail, n, p, d, dh in zip(
            batch, heads, tails, ner, pos, dep, dep_head
        ):
            inputs.append(
                dict(
                    text=" ".join(sent),
                    head=head,
                    tail=tail,
                    ner=n,
                    pos=p,
                    dep=d,
                    dep_heads=dh,
                )
            )
        results = predictor.predict_batch_json(inputs)
        sent_embeddings = np.array([result["input_rep"] for result in results])
        return sent_embeddings

    predictor = load_predictor(
        args.model_dir,
        args.predictor,
        args.cuda_device,
        archive_filename="model.tar.gz",
        weights_file=None,
    )

    if args.prototyping:
        params = {
            "task_path": args.data_dir,
            "usepytorch": True,
            "kfold": 5,
            "batch_size": args.batch_size,
        }
        params["classifier"] = {
            "nhid": 0,
            "optim": "rmsprop",
            "batch_size": 128,
            "tenacity": 3,
            "epoch_size": 2,
        }
    else:
        params = {
            "task_path": args.data_dir,
            "usepytorch": True,
            "kfold": 10,
            "batch_size": args.batch_size,
        }
        params["classifier"] = {
            "nhid": 0,
            "optim": "adam",
            "batch_size": 64,
            "tenacity": 5,
            "epoch_size": 4,
        }

    tasks = ALL_PROBING_TASKS

    logger.info(f"Parameters: {json.dumps(params, indent=4, sort_keys=True)}")
    logger.info(f"Tasks: {tasks}")

    re = reval.engine.RE(params, batcher, prepare)
    results = re.eval(tasks)

    logger.info(
        f"Probing Task Results: {json.dumps(results, indent=4, sort_keys=True)}"
    )

    output_dir = args.output_dir or args.model_dir

    with open(os.path.join(output_dir, "probing_task_results.json"), "w") as prob_res_f:
        json.dump(results, prob_res_f, indent=4, sort_keys=True)


if __name__ == "__main__":
    main()
