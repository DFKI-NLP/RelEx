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
            "ner": [
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
                "PERSON",
                "PERSON",
                "O",
                "O",
                "O",
                "O",
                "O",
                "PERSON",
                "PERSON",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
            ],
            "pos": [
                "IN",
                "DT",
                "JJ",
                "NN",
                ",",
                "NNP",
                "NNP",
                "NNP",
                "NNP",
                "NNP",
                "MD",
                "VB",
                "NN",
                ",",
                "VBG",
                "NNP",
                "NNP",
                "WP",
                "VBZ",
                "VBG",
                "TO",
                "VB",
                "DT",
                "NN",
                "NN",
                ".",
            ],
        }

        assert len(instances) == 3

        fields = instances[0].fields

        tokens = fields["text"].tokens
        assert [t.text for t in tokens] == instance1["tokens"]
        assert [t.ent_type_ for t in tokens] == instance1["ner"]
        assert [t.tag_ for t in tokens] == instance1["pos"]

        assert fields["label"].label == instance1["label"]
        assert fields["head"].span_start == instance1["head"][0]
        assert fields["head"].span_end == instance1["head"][1]
        assert fields["tail"].span_start == instance1["tail"][0]
        assert fields["tail"].span_end == instance1["tail"][1]
        assert fields["metadata"]["id"] == instance1["id"]

    def test_ner_masking(self):
        MAX_LEN = 100
        reader = TacredDatasetReader(max_len=MAX_LEN, masking_mode="NER")
        instances = ensure_list(reader.read("tests/fixtures/tacred.json"))

        expected_tokens = [
            "At",
            "the",
            "same",
            "time",
            ",",
            "Chief",
            "Financial",
            "Officer",
            "__PERSON__",
            "__PERSON__",
            "will",
            "become",
            "__TITLE__",
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
        ]

        tokens = instances[0].fields["text"].tokens
        assert [t.text for t in tokens] == expected_tokens

    def test_ner_grammar_masking(self):
        MAX_LEN = 100
        reader = TacredDatasetReader(max_len=MAX_LEN, masking_mode="NER+Grammar")
        instances = ensure_list(reader.read("tests/fixtures/tacred.json"))

        expected_tokens = [
            "At",
            "the",
            "same",
            "time",
            ",",
            "Chief",
            "Financial",
            "Officer",
            "__PERSON_SUB__",
            "__PERSON_SUB__",
            "will",
            "become",
            "__TITLE_OBJ__",
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
        ]

        tokens = instances[0].fields["text"].tokens
        assert [t.text for t in tokens] == expected_tokens

    def test_grammar_masking(self):
        MAX_LEN = 100
        reader = TacredDatasetReader(max_len=MAX_LEN, masking_mode="Grammar")
        instances = ensure_list(reader.read("tests/fixtures/tacred.json"))

        expected_tokens = [
            "At",
            "the",
            "same",
            "time",
            ",",
            "Chief",
            "Financial",
            "Officer",
            "__SUB__",
            "__SUB__",
            "will",
            "become",
            "__OBJ__",
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
        ]

        tokens = instances[0].fields["text"].tokens
        assert [t.text for t in tokens] == expected_tokens

    def test_unknown_masking(self):
        MAX_LEN = 100
        reader = TacredDatasetReader(max_len=MAX_LEN, masking_mode="UNK")
        instances = ensure_list(reader.read("tests/fixtures/tacred.json"))

        expected_tokens = [
            "At",
            "the",
            "same",
            "time",
            ",",
            "Chief",
            "Financial",
            "Officer",
            "__UNK__",
            "__UNK__",
            "will",
            "become",
            "__UNK__",
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
        ]

        tokens = instances[0].fields["text"].tokens
        assert [t.text for t in tokens] == expected_tokens
