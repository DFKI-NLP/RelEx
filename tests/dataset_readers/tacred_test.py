from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

from relex.dataset_readers import TacredDatasetReader


class TestTacredDatasetReader(AllenNlpTestCase):
    def test_read_from_file(self):
        MAX_LEN = 100

        reader = TacredDatasetReader(max_len=MAX_LEN)
        instances = ensure_list(reader.read("tests/fixtures/tacred.json"))

        instance1 = {
            "tokens": [
                "At",
                "the",
                "same",
                "time",
                ",",
                "Chief",
                "Financial",
                "Officer",
                "Douglas",
                "Flint",
                "will",
                "become",
                "chairman",
                ",",
                "succeeding",
                "Stephen",
                "Green",
                "who",
                "is",
                "leaving",
                "to",
                "take",
                "a",
                "government",
                "job",
                ".",
            ],
            "head": (8, 9),
            "tail": (12, 12),
            "id": "e7798fb926b9403cfcd2",
            "label": "per:title",
        }

        assert len(instances) == 3
        fields = instances[0].fields
        tokens = fields["text"].tokens
        assert [t.text for t in tokens] == instance1["tokens"]
        assert fields["label"].label == instance1["label"]
        assert fields["head"].span_start == instance1["head"][0]
        assert fields["head"].span_end == instance1["head"][1]
        assert fields["tail"].span_start == instance1["tail"][0]
        assert fields["tail"].span_end == instance1["tail"][1]
        assert fields["metadata"]["id"] == instance1["id"]
