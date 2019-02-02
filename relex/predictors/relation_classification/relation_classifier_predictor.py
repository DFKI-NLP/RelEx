from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register("relation_classifier")
class RelationClassifierPredictor(Predictor):
    """"Predictor wrapper for the RelationClassifier"""

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        instance = self._dataset_reader.text_to_instance(**json_dict)
        return instance
