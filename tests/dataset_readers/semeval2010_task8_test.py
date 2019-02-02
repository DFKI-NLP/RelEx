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

        head_offsets = [
            -12,
            -11,
            -10,
            -9,
            -8,
            -7,
            -6,
            -5,
            -4,
            -3,
            -2,
            -1,
            0,
            1,
            2,
            3,
            4,
        ]
        head_offsets = [o + MAX_LEN for o in head_offsets]

        tail_offsets = [
            -15,
            -14,
            -13,
            -12,
            -11,
            -10,
            -9,
            -8,
            -7,
            -6,
            -5,
            -4,
            -3,
            -2,
            -1,
            0,
            1,
        ]
        tail_offsets = [o + MAX_LEN for o in tail_offsets]

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

        assert all([hasattr(token, "offset_head") for token in tokens])
        assert all([hasattr(token, "offset_tail") for token in tokens])

        assert head_offsets == [token.offset_head for token in tokens]
        assert tail_offsets == [token.offset_tail for token in tokens]
