# pylint: disable=no-self-use,invalid-name,unused-import
from unittest import TestCase

from pytest import approx
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

import relex


class RelationClassifierPredictorTest(TestCase):
    def test_uses_named_inputs(self):
        inputs = {
            "text": (
                "The system as described above has its greatest "
                "application in an arrayed configuration of antenna elements ."
            ),
            "head": [12, 13],
            "tail": [15, 16],
        }

        archive = load_archive("tests/fixtures/model.tar.gz")
        predictor = Predictor.from_archive(archive, "relation_classifier")

        result = predictor.predict_json(inputs)

        label = result.get("label")
        assert label in {
            "Other",
            "Entity-Destination(e1,e2)",
            "Component-Whole(e2,e1)",
            "Instrument-Agency(e2,e1)",
            "Member-Collection(e1,e2)",
            "Cause-Effect(e2,e1)",
            "Content-Container(e1,e2)",
        }

        class_probabilities = result.get("class_probabilities")
        assert class_probabilities is not None
        assert all(cp > 0 for cp in class_probabilities)
        assert sum(class_probabilities) == approx(1.0)
