from typing import Optional, Dict, Set
from collections import defaultdict

import torch

from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics.metric import Metric
from allennlp.common.checks import ConfigurationError


class F1Measure(Metric):
    """
    Computes Precision, Recall and F1 with respect to a given ``positive_label``.
    For example, for a BIO tagging scheme, you would pass the classification index of
    the tag you are interested in, resulting in the Precision, Recall and F1 score being
    calculated for this tag only.
    """

    def __init__(
        self,
        vocabulary: Vocabulary,
        average: str = "macro",
        label_namespace: str = "labels",
        ignore_label: str = None,
    ) -> None:
        self._label_vocabulary = vocabulary.get_index_to_token_vocabulary(
            label_namespace
        )
        self._average = average
        self._ignore_label = ignore_label
        self._true_positives: Dict[str, int] = defaultdict(int)
        self._true_negatives: Dict[str, int] = defaultdict(int)
        self._false_positives: Dict[str, int] = defaultdict(int)
        self._false_negatives: Dict[str, int] = defaultdict(int)

    def __call__(
        self,
        predictions: torch.Tensor,
        gold_labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, ..., num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the ``predictions`` tensor without the ``num_classes`` dimension.
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        """
        predictions, gold_labels, mask = self.unwrap_to_tensors(
            predictions, gold_labels, mask
        )

        num_classes = predictions.size(-1)
        if (gold_labels >= num_classes).any():
            raise ConfigurationError(
                "A gold label passed to F1Measure contains an id >= {}, "
                "the number of classes.".format(num_classes)
            )
        if mask is None:
            mask = torch.ones_like(gold_labels)
        mask = mask.float()
        gold_labels = gold_labels.float()

        argmax_predictions = predictions.max(-1)[1].float().squeeze(-1)

        for label_index, label_token in self._label_vocabulary.items():

            positive_label_mask = gold_labels.eq(label_index).float()
            negative_label_mask = 1.0 - positive_label_mask

            # True Negatives: correct non-positive predictions.
            correct_null_predictions = (
                argmax_predictions != label_index
            ).float() * negative_label_mask
            self._true_negatives[label_token] += (
                correct_null_predictions.float() * mask
            ).sum()

            # True Positives: correct positively labeled predictions.
            correct_non_null_predictions = (
                argmax_predictions == label_index
            ).float() * positive_label_mask
            self._true_positives[label_token] += (
                correct_non_null_predictions * mask
            ).sum()

            # False Negatives: incorrect negatively labeled predictions.
            incorrect_null_predictions = (
                argmax_predictions != label_index
            ).float() * positive_label_mask
            self._false_negatives[label_token] += (
                incorrect_null_predictions * mask
            ).sum()

            # False Positives: incorrect positively labeled predictions
            incorrect_non_null_predictions = (
                argmax_predictions == label_index
            ).float() * negative_label_mask
            self._false_positives[label_token] += (
                incorrect_non_null_predictions * mask
            ).sum()

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        A tuple of the following metrics based on the accumulated count statistics:
        precision : float
        recall : float
        f1-measure : float
        """
        all_tags: Set[str] = set()
        all_tags.update(self._true_positives.keys())
        all_tags.update(self._false_positives.keys())
        all_tags.update(self._false_negatives.keys())
        all_metrics = {}

        for tag in all_tags:
            precision, recall, f1_measure = self._compute_metrics(
                self._true_positives[tag],
                self._false_positives[tag],
                self._false_negatives[tag],
            )
            precision_key = "precision" + "-" + tag
            recall_key = "recall" + "-" + tag
            f1_key = "f1-measure" + "-" + tag
            all_metrics[precision_key] = precision
            all_metrics[recall_key] = recall
            all_metrics[f1_key] = f1_measure

        if self._average == "micro":
            if self._ignore_label is not None:
                precision, recall, f1_measure = self._compute_metrics(
                    sum(
                        [
                            val
                            for l, val in self._true_positives.items()
                            if l != self._ignore_label
                        ]
                    ),
                    sum(
                        [
                            val
                            for l, val in self._false_positives.items()
                            if l != self._ignore_label
                        ]
                    ),
                    sum(
                        [
                            val
                            for l, val in self._false_negatives.items()
                            if l != self._ignore_label
                        ]
                    ),
                )
            else:
                precision, recall, f1_measure = self._compute_metrics(
                    sum(self._true_positives.values()),
                    sum(self._false_positives.values()),
                    sum(self._false_negatives.values()),
                )
        elif self._average == "macro":
            precision = 0.0
            recall = 0.0
            n_precision = 0
            n_recall = 0

            for tag in all_tags:
                precision_key = "precision" + "-" + tag
                recall_key = "recall" + "-" + tag
                precision += all_metrics[precision_key]
                recall += all_metrics[recall_key]
                n_precision += 1
                n_recall += 1

            if n_precision:
                precision /= n_precision
            if n_recall:
                recall /= n_recall
            f1_measure = 2.0 * ((precision * recall) / (precision + recall + 1e-13))

        all_metrics["precision-overall"] = precision
        all_metrics["recall-overall"] = recall
        all_metrics["f1-measure-overall"] = f1_measure
        if reset:
            self.reset()
        return all_metrics

    @staticmethod
    def _compute_metrics(
        true_positives: int, false_positives: int, false_negatives: int
    ):
        precision = float(true_positives) / float(
            true_positives + false_positives + 1e-13
        )
        recall = float(true_positives) / float(true_positives + false_negatives + 1e-13)
        f1_measure = 2.0 * ((precision * recall) / (precision + recall + 1e-13))
        return precision, recall, f1_measure

    def reset(self):
        self._true_positives = defaultdict(int)
        self._true_negatives = defaultdict(int)
        self._false_positives = defaultdict(int)
        self._false_negatives = defaultdict(int)
