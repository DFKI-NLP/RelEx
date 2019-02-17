from typing import Dict, Optional, List, Any

import numpy
from allennlp.data.vocabulary import DEFAULT_OOV_TOKEN
from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy
from torch.distributions import Bernoulli

from relex.modules.offset_embedders import OffsetEmbedder
from relex.metrics import F1Measure


@Model.register("basic_relation_classifier")
class BasicRelationClassifier(Model):
    """
    This ``Model`` performs relation classification on a given input. 
    We assume we're given a text, head and tail entity offsets, and we predict the 
    relation between head and tail entity. The basic model structure: we'll embed the 
    text, relative head and tail offsets, and encode it with a Seq2VecEncoder, getting a 
    single vector representing the content.  We'll then pass the result through a 
    feedforward network, the output of which we'll use as our scores for each label.
    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    text_encoder : ``Seq2VecEncoder``
        The encoder that we will use to convert the text to a vector.
    classifier_feedforward : ``FeedForward``
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    offset_embedder_head : ``OffsetEmbedder``
        The embedder we use to embed each tokens relative offset to the head entity.
    offset_embedder_tail : ``OffsetEmbedder``
        The embedder we use to embed each tokens relative offset to the tail entity.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        text_encoder: Seq2VecEncoder,
        classifier_feedforward: FeedForward,
        word_dropout: Optional[float] = None,
        encoding_dropout: Optional[float] = None,
        offset_embedder_head: Optional[OffsetEmbedder] = None,
        offset_embedder_tail: Optional[OffsetEmbedder] = None,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
        verbose_metrics: bool = False,
        ignore_label: str = None,
        f1_average: str = "macro",
    ) -> None:
        super(BasicRelationClassifier, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.text_encoder = text_encoder
        self.word_dropout = word_dropout
        self.encoding_dropout = (
            torch.nn.Dropout(encoding_dropout) if encoding_dropout else None
        )
        self.classifier_feedforward = classifier_feedforward
        self.offset_embedder_head = offset_embedder_head
        self.offset_embedder_tail = offset_embedder_tail
        self._verbose_metrics = verbose_metrics

        offset_embedding_dim = 0
        if offset_embedder_head is not None:
            if not offset_embedder_head.is_additive():
                offset_embedding_dim += offset_embedder_head.get_output_dim()

        if offset_embedder_tail is not None:
            if not offset_embedder_tail.is_additive():
                offset_embedding_dim += offset_embedder_tail.get_output_dim()

        text_encoder_input_dim = (
            text_field_embedder.get_output_dim() + offset_embedding_dim
        )

        if text_encoder_input_dim != text_encoder.get_input_dim():
            raise ConfigurationError(
                "The output dimension of the text_field_embedder and offset_embedders "
                "must match the input dimension of the text_encoder. Found {} and {}, "
                "respectively.".format(
                    text_field_embedder.get_output_dim(), text_encoder.get_input_dim()
                )
            )

        if text_encoder.get_output_dim() != classifier_feedforward.get_input_dim():
            raise ConfigurationError(
                "The output dimension of the text_encoder must match the "
                "input dimension of the classifier_feedforward. Found {} and {}, "
                "respectively.".format(
                    text_encoder.get_output_dim(),
                    classifier_feedforward.get_input_dim(),
                )
            )

        if classifier_feedforward.get_output_dim() != self.num_classes:
            raise ConfigurationError(
                "The output dimension of the classifier_feedforward must match the "
                "number of classes in the dataset. Found {} and {}, "
                "respectively.".format(
                    classifier_feedforward.get_output_dim(), self.num_classes
                )
            )

        self.metrics = {"accuracy": CategoricalAccuracy()}
        self._f1_measure = F1Measure(
            vocabulary=self.vocab, average=f1_average, ignore_label=ignore_label
        )

        self.loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    @overrides
    def forward(
        self,  # type: ignore
        text: Dict[str, torch.LongTensor],
        head: torch.LongTensor,
        tail: torch.LongTensor,
        metadata: Optional[List[Dict[str, Any]]] = None,
        label: Optional[torch.LongTensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        text : Dict[str, Variable], required
            The output of ``TextField.as_array()``.
        label : Variable, optional (default = None)
            A variable representing the label for each instance in the batch.
        Returns
        -------
        An output dictionary consisting of:
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_classes)`` representing a distribution over the
            label classes for each instance.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        text_mask = util.get_text_field_mask(text)

        if self.training and self.word_dropout is not None:
            # Generate a binary mask with ones for dropped words
            dropout_mask = Bernoulli(self.word_dropout).sample(text_mask.shape)
            dropout_mask = dropout_mask.to(device=text_mask.device)
            dropout_mask = dropout_mask.byte() & text_mask.byte()

            # Set the dropped words to the OOV token index
            oov_token_idx = self.vocab.get_token_index(DEFAULT_OOV_TOKEN)
            text['tokens'].masked_fill_(dropout_mask, oov_token_idx)

        embedded_text = self.text_field_embedder(text)

        embeddings = [embedded_text]
        if self.offset_embedder_head is not None:
            embeddings.append(
                self.offset_embedder_head(embedded_text, text_mask, span=head)
            )
        if self.offset_embedder_tail is not None:
            embeddings.append(
                self.offset_embedder_tail(embedded_text, text_mask, span=tail)
            )

        if len(embeddings) > 1:
            embedded_text = torch.cat(embeddings, dim=-1)
        else:
            embedded_text = embeddings[0]

        encoded_text = self.text_encoder(embedded_text, text_mask)

        if self.encoding_dropout is not None:
            encoded_text = self.encoding_dropout(encoded_text)

        logits = self.classifier_feedforward(encoded_text)

        output_dict = {"logits": logits, "input_rep": encoded_text}
        if label is not None:
            loss = self.loss(logits, label)
            for metric in self.metrics.values():
                metric(logits, label)
            self._f1_measure(logits, label)
            output_dict["loss"] = loss

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a ``"label"`` key to the dictionary with the result.
        """
        class_probabilities = F.softmax(output_dict["logits"], dim=-1)
        output_dict["class_probabilities"] = class_probabilities

        predictions = class_probabilities.cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
        labels = [
            self.vocab.get_token_from_index(x, namespace="labels")
            for x in argmax_indices
        ]
        output_dict["label"] = labels
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {
            metric_name: metric.get_metric(reset)
            for metric_name, metric in self.metrics.items()
        }

        f1_dict = self._f1_measure.get_metric(reset=reset)
        if self._verbose_metrics:
            metrics_to_return.update(f1_dict)
        else:
            metrics_to_return.update(
                {x: y for x, y in f1_dict.items() if "overall" in x}
            )

        return metrics_to_return
