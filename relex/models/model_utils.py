from typing import Dict, List, Any

from allennlp.predictors import Predictor
from allennlp.data import Instance


def batched_predict_json(
        predictor: Predictor,
        examples: List[Dict[str, Any]],
        batch_size: int = 16) -> List[Dict[str, Any]]:
    results = []  # type: List[Dict[str, Any]]
    for i in range(0, len(examples), batch_size):
        batch_examples = examples[i: i + batch_size]
        batch_results = predictor.predict_batch_json(batch_examples)
        results.extend(batch_results)
    return results


def batched_predict_instances(
        predictor: Predictor,
        examples: List[Instance],
        batch_size: int = 16) -> List[Dict[str, Any]]:
    results = []  # type: List[Dict[str, Any]]
    for i in range(0, len(examples), batch_size):
        batch_examples = examples[i: i + batch_size]
        batch_results = predictor.predict_batch_instance(batch_examples)
        results.extend(batch_results)
    return results
