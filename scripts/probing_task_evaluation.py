from typing import Optional

import os
import logging
import argparse
import json
import numpy as np
import relex
import reval
from relex.predictors.utils import load_predictor

logger = logging.getLogger()
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
        "--cuda-device", type=int, default=0, help="a cuda device to load the model on"
    )
    parser.add_argument("--prototyping", action="store_true")

    parser.add_argument("--cache-representations", action="store_true")

    return parser


def run_evaluation(
    model_dir: str,
    data_dir: str,
    output_dir: Optional[str] = None,
    predictor: str = "relation_classifier",
    batch_size: int = 128,
    cuda_device: int = 0,
    prototyping: bool = False,
    cache_representations: bool = True,
):

    predictor = load_predictor(
        model_dir,
        predictor,
        cuda_device,
        archive_filename="model.tar.gz",
        weights_file=None,
    )

    def prepare(params, samples):
        pass

    cache = {}

    def batcher(params, batch, heads, tails, ner, pos, dep, dep_head, ids):
        if cache_representations:
            inputs = []
            inputs_ids = []

            for sent, head, tail, n, p, d, dh, id_ in zip(
                batch, heads, tails, ner, pos, dep, dep_head, ids
            ):
                if id_ not in cache:
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
                    inputs_ids.append(id_)

            if inputs:
                computed_sent_embeddings = {
                    id_: result["input_rep"]
                    for id_, result in zip(
                        inputs_ids, predictor.predict_batch_json(inputs)
                    )
                }
                cache.update(computed_sent_embeddings)

            sent_embeddings = np.array([cache[id_] for id_ in ids])

        else:
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

    if prototyping:
        params = {
            "task_path": data_dir,
            "usepytorch": True,
            "kfold": 5,
            "batch_size": batch_size,
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
            "task_path": data_dir,
            "usepytorch": True,
            "kfold": 10,
            "batch_size": batch_size,
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

    output_dir = output_dir or model_dir

    with open(os.path.join(output_dir, "probing_task_results.json"), "w") as prob_res_f:
        json.dump(results, prob_res_f, indent=4, sort_keys=True)


if __name__ == "__main__":
    parser = _get_parser()
    args = parser.parse_args()
    run_evaluation(
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        predictor=args.predictor,
        batch_size=args.batch_size,
        cuda_device=args.cuda_device,
        prototyping=args.prototyping,
        cache_representations=args.cache_representations,
    )
