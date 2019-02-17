# pylint: disable=invalid-name,protected-access
from allennlp.common.testing import ModelTestCase


class BasicRelationClassifierTest(ModelTestCase):
    def setUp(self):
        super(BasicRelationClassifierTest, self).setUp()
        self.set_up_model(
            "tests/fixtures/basic_relation_classifier.jsonnet",
            "tests/fixtures/semeval2010_task8.jsonl",
        )

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
