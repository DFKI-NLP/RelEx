from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

from relex.dataset_readers import SemEval2010Task8DatasetReader


class TestSemEval2010Task8DatasetReader(AllenNlpTestCase):
    def test_read_from_file(self):
        MAX_LEN = 100

        reader = SemEval2010Task8DatasetReader(max_len=MAX_LEN)
        instances = ensure_list(reader.read("tests/fixtures/semeval2010_task8.jsonl"))

        instance1 = {
            "tokens": [
                "The",
                "system",
                "as",
                "described",
                "above",
                "has",
                "its",
                "greatest",
                "application",
                "in",
                "an",
                "arrayed",
                "configuration",
                "of",
                "antenna",
                "elements",
                ".",
            ],
            "head": (12, 12),
            "tail": (15, 15),
            "id": "1",
            "label": "Component-Whole(e2,e1)",
        }

        assert len(instances) == 10
        fields = instances[0].fields
        tokens = fields["text"].tokens
        assert [t.text for t in tokens] == instance1["tokens"]
        assert fields["label"].label == instance1["label"]
        assert fields["head"].span_start == instance1["head"][0]
        assert fields["head"].span_end == instance1["head"][1]
        assert fields["tail"].span_start == instance1["tail"][0]
        assert fields["tail"].span_end == instance1["tail"][1]
        assert fields["metadata"]["id"] == instance1["id"]

    def test_max_len(self):
        MAX_LEN = 13

        reader = SemEval2010Task8DatasetReader(max_len=MAX_LEN)
        instances = ensure_list(reader.read("tests/fixtures/semeval2010_task8.jsonl"))

        fields = instances[0].fields
        tokens = fields["text"].tokens
        head_span = (fields["head"].span_start, fields["head"].span_end)
        tail_span = (fields["tail"].span_start, fields["tail"].span_end)

        assert [t.text for t in tokens] == [
            "The",
            "system",
            "as",
            "described",
            "above",
            "has",
            "its",
            "greatest",
            "application",
            "in",
            "an",
            "arrayed",
            "configuration",
        ]
        assert head_span == (12, 12)
        assert tail_span == (12, 12)
