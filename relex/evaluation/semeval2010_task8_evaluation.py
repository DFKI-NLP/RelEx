from typing import List, Optional

import csv
from subprocess import run, PIPE
from tempfile import NamedTemporaryFile
from relex.predictors.utils import load_predictor
from relex.models.utils import batched_predict_instances


def _write_id_label_file(file_path: str, ids: List[str], labels: List[str]) -> None:
    assert len(ids) == len(labels)

    with open(file_path, "w") as f:
        writer = csv.writer(f, delimiter="\t")
        for row in zip(ids, labels):
            writer.writerow(row)


def evaluate(
    model_dir: str,
    test_file: str,
    eval_script_file,
    archive_filename: str = "model.tar.gz",
    cuda_device: int = -1,
    predictor_name: str = "relation_classifier",
    weights_file: Optional[str] = None,
    batch_size: int = 16,
) -> str:
    # load predictor from archive in model dir
    # load test file via dataset loader from test file
    # predict all instances
    # compute and output official evaluation script result

    predictor = load_predictor(
        model_dir, predictor_name, cuda_device, archive_filename, weights_file
    )

    test_instances = predictor._dataset_reader.read(test_file)
    test_results = batched_predict_instances(predictor, test_instances, batch_size)

    instance_ids = [instance["metadata"]["id"] for instance in test_instances]
    true_labels = [instance["label"].label for instance in test_instances]
    predicted_labels = [result["label"] for result in test_results]

    true_labels_file = NamedTemporaryFile(delete=True).name
    _write_id_label_file(true_labels_file, instance_ids, true_labels)
    predicted_labels_file = NamedTemporaryFile(delete=True).name
    _write_id_label_file(predicted_labels_file, instance_ids, predicted_labels)

    p = run(
        [eval_script_file, true_labels_file, predicted_labels_file],
        stdout=PIPE,
        encoding="utf-8",
    )

    report = p.stdout
    return report
