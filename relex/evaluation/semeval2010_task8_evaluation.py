from typing import List, Optional

import re
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


PRECISION_REGEX = r"P =\s*([0-9]{1,2}\.[0-9]{2})%"
RECALL_REGEX = r"R =\s*([0-9]{1,2}\.[0-9]{2})%"
F1_REGEX = r"F1 =\s*([0-9]{1,2}\.[0-9]{2})%"

OFFICIAL_RESULT_REGEX = (
    r"\(9\+1\)-WAY EVALUATION TAKING DIRECTIONALITY INTO ACCOUNT -- OFFICIAL"
)
RESULT_LINE_REGEX = r"MACRO-averaged result \(excluding Other\):\n((.*\n){1})"


def prec_rec_f1_from_report(report):
    official_result_match = re.search(OFFICIAL_RESULT_REGEX, report)

    if official_result_match:
        result_start = official_result_match.span(0)[1]
        match = re.search(RESULT_LINE_REGEX, report[result_start:])

        precision = None
        recall = None
        f1 = None
        if match:
            result_line = match.group(1)
            precision_match = re.search(PRECISION_REGEX, result_line)
            recall_match = re.search(RECALL_REGEX, result_line)
            f1_match = re.search(F1_REGEX, result_line)

            if precision_match:
                precision = float(precision_match.group(1))
            if recall_match:
                recall = float(recall_match.group(1))
            if f1_match:
                f1 = float(f1_match.group(1))

    return precision, recall, f1


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
    print(report)
    return prec_rec_f1_from_report(report)
