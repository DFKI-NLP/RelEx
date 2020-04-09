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
            "dep": [
                "case",
                "det",
                "amod",
                "nmod",
                "punct",
                "compound",
                "compound",
                "compound",
                "compound",
                "nsubj",
                "aux",
                "ROOT",
                "xcomp",
                "punct",
                "xcomp",
                "compound",
                "dobj",
                "nsubj",
                "aux",
                "acl:relcl",
                "mark",
                "xcomp",
                "det",
                "compound",
                "dobj",
                "punct",
            ],
        }

        assert len(instances) == 3

        fields = instances[0].fields

        tokens = fields["text"].tokens
        assert [t.text for t in tokens] == instance1["tokens"]
        print([t.ent_type_ for t in tokens])
        assert [t.ent_type_ for t in tokens] == instance1["ner"]
        assert [t.tag_ for t in tokens] == instance1["pos"]
        assert [t.dep_ for t in tokens] == instance1["dep"]

        assert fields["label"].label == instance1["label"]
        assert fields["head"].span_start == instance1["head"][0]
        assert fields["head"].span_end == instance1["head"][1]
        assert fields["tail"].span_start == instance1["tail"][0]
        assert fields["tail"].span_end == instance1["tail"][1]
        assert fields["metadata"]["id"] == instance1["id"]

    def test_max_len(self):
        MAX_LEN = 10
        reader = TacredDatasetReader(max_len=MAX_LEN, masking_mode="NER")
        instances = ensure_list(reader.read("tests/fixtures/tacred.json"))

        fields = instances[0].fields
        tokens = fields["text"].tokens
        head_span = (fields["head"].span_start, fields["head"].span_end)
        tail_span = (fields["tail"].span_start, fields["tail"].span_end)

        assert [t.text for t in tokens] == [
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
        ]
        assert head_span == (8, 9)
        assert tail_span == (9, 9)

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

    def test_unpruned_adjacency_matrix(self):
        MAX_LEN = 100
        reader = TacredDatasetReader(max_len=MAX_LEN,
                                     masking_mode="NER",
                                     dep_pruning=-1)
        instances = ensure_list(reader.read("tests/fixtures/tacred.json"))

        expected_edges = [
            (0, 0), (0, 3), (1, 1), (1, 3), (2, 2), (2, 3), (3, 0), (3, 1),
            (3, 2), (3, 3), (3, 11), (4, 4), (4, 11), (5, 5), (5, 9), (6, 6),
            (6, 9), (7, 7), (7, 9), (8, 8), (8, 9), (9, 5), (9, 6), (9, 7),
            (9, 8), (9, 9), (9, 11), (10, 10), (10, 11), (11, 3), (11, 4),
            (11, 9), (11, 10), (11, 11), (11, 12), (11, 13), (11, 14),
            (11, 25), (12, 11), (12, 12), (13, 11), (13, 13), (14, 11),
            (14, 14), (14, 16), (15, 15), (15, 16), (16, 14), (16, 15),
            (16, 16), (16, 19), (17, 17), (17, 19), (18, 18), (18, 19),
            (19, 16), (19, 17), (19, 18), (19, 19), (19, 21), (20, 20),
            (20, 21), (21, 19), (21, 20), (21, 21), (21, 24), (22, 22),
            (22, 24), (23, 23), (23, 24), (24, 21), (24, 22), (24, 23),
            (24, 24), (25, 11), (25, 25)
        ]
        adjacency = instances[0].fields["adjacency"]

        assert sorted(adjacency.indices) == expected_edges

    def test_k1_pruned_adjacency_matrix(self):
        MAX_LEN = 100
        reader = TacredDatasetReader(max_len=MAX_LEN,
                                     masking_mode="NER",
                                     dep_pruning=1)
        instances = ensure_list(reader.read("tests/fixtures/tacred.json"))

        expected_edges = [
            (3, 3), (3, 11), (4, 4), (4, 11), (5, 5), (5, 9), (6, 6), (6, 9),
            (7, 7), (7, 9), (8, 8), (8, 9), (9, 5), (9, 6), (9, 7), (9, 8),
            (9, 9), (9, 11), (10, 10), (10, 11), (11, 3), (11, 4), (11, 9),
            (11, 10), (11, 11), (11, 12), (11, 13), (11, 14), (11, 25),
            (12, 11), (12, 12), (13, 11), (13, 13), (14, 11), (14, 14),
            (25, 11), (25, 25)
        ]
        adjacency = instances[0].fields["adjacency"]

        assert sorted(adjacency.indices) == expected_edges
