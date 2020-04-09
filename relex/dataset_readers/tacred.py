from typing import Dict, List, Tuple, Union, Optional
import json
import logging

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import (LabelField, TextField, SpanField,
                                  MetadataField, AdjacencyField)
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

from relex.dataset_readers.dataset_reader_utils import parse_adjacency_indices

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def normalize_glove(token):
    mapping = {
            "-LRB-": "(",
            "-RRB-": ")",
            "-LSB-": "[",
            "-RSB-": "]",
            "-LCB-": "{",
            "-RCB-": "}",
    }
    return mapping.get(token, token)


@DatasetReader.register("tacred")
class TacredDatasetReader(DatasetReader):
    """
    Reads a JSON file containing examples in TACRED format and creates a
    dataset suitable for relation classification. The expected keys per
    example are "token", "relation", "{subj, obj}_start", "{subj, obj}_end"
    and "{subj, obj}_type". Optional keys are "stanford_ner", "stanford_pos",
    "stanford_head" and "stanford_deprel".

    Parameters
    ----------
    max_len : ``int``, required
        Max number of tokens for each text. This is necessary for computing the relative offset
        of head and tail entities.
    masking_mode : ``str``, optional
        Masking mode to replace the head and tail entity tokens. "NER" replaces each entity with
        a special tag consisting of its named entity tag, "Grammar" uses its grammatical role,
        "NER+Grammar" combines both, "UNK" replaces the entity with the unknown token, and
        "NER_NL" uses the actual named entity word, e.g. "organization".
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
        take longer per batch.  This also allows training with datasets that are too large to fit
        in memory.
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the text into words or other kinds of tokens.
        Defaults to ``WordTokenizer(word_splitter=JustSpacesWordSplitter())``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input token representations. Defaults to ``{"tokens":
        SingleIdTokenIndexer()}``.
    dep_pruning : ``int``, (optional, default=1)
        If >= 0 and dependency parse information is provided, prune the tree along the tokens
        on the shortest dependency path (SDP) between head and tail entity, while retaining all
        other tokens within "dep_pruning" distance to any of the SDP nodes.
    """

    def __init__(self,
                 max_len: int,
                 masking_mode: str = None,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 dep_pruning: int = 1) -> None:
        super().__init__(lazy)
        self._max_len = max_len
        self._tokenizer = (tokenizer
                           or WordTokenizer(word_splitter=JustSpacesWordSplitter()))
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._masking_mode = masking_mode
        self._dep_pruning = dep_pruning

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r") as data_file:
            logger.info("Reading TACRED instances from json dataset at: %s", file_path)
            data = json.load(data_file)
            for example in data:
                tokens = example["token"]
                head = (example["subj_start"], example["subj_end"])
                tail = (example["obj_start"], example["obj_end"])
                head_type = example["subj_type"]
                tail_type = example["obj_type"]
                relation = example["relation"]

                id_ = example.get("id")
                ner = example.get("stanford_ner")
                pos = example.get("stanford_pos")
                dep = example.get("stanford_deprel")
                dep_heads = example.get("stanford_head")

                if self._masking_mode is not None:
                    tokens = self._apply_masking_mode(tokens,
                                                      head,
                                                      tail,
                                                      head_type,
                                                      tail_type)

                yield self.text_to_instance(tokens, head, tail, id_,
                                            relation, ner, pos, dep, dep_heads)

    @overrides
    def text_to_instance(self,
                         text: Union[str, List[str]],
                         head: Tuple[int, int],
                         tail: Tuple[int, int],
                         id_: Optional[str] = None,
                         relation: Optional[str] = None,
                         ner: Optional[List[str]] = None,
                         pos: Optional[List[str]] = None,
                         dep: Optional[List[str]] = None,
                         dep_heads: Optional[List[int]] = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ

        tokenized_text: List[Token] = []
        if isinstance(text, str):
            tokenized_text = self._tokenizer.tokenize(text)
        else:
            for idx, token_text in enumerate(text):
                tokenized_text.append(Token(normalize_glove(token_text),  # text
                                            idx,  # idx
                                            None,  # lemma_
                                            pos[idx] if pos is not None else None,  # pos_
                                            pos[idx] if pos is not None else None,  # tag_
                                            dep[idx] if dep is not None else None,  # dep_
                                            ner[idx] if ner is not None else None))  # ent_type_
        
        head_start, head_end = head
        tail_start, tail_end = tail

        # make sure head and tail span stays within max input length
        head_start = min(head_start, self._max_len - 1)
        head_end = min(head_end, self._max_len - 1)
        tail_start = min(tail_start, self._max_len - 1)
        tail_end = min(tail_end, self._max_len - 1)

        text_tokens_field = TextField(tokenized_text[: self._max_len],
                                      self._token_indexers)

        fields = {
                "text": text_tokens_field,
                "head": SpanField(head_start, head_end, sequence_field=text_tokens_field),
                "tail": SpanField(tail_start, tail_end, sequence_field=text_tokens_field),
        }

        if dep is not None and dep_heads is not None:
            indices = parse_adjacency_indices(dep, dep_heads, head, tail,
                                              pruning_distance=self._dep_pruning)

            # Only keep edges within the clipped sentence length
            indices = [idx_pair for idx_pair in indices
                       if idx_pair[0] < self._max_len and idx_pair[1] < self._max_len]

            fields["adjacency"] = AdjacencyField(indices,
                                                 sequence_field=text_tokens_field,
                                                 padding_value=0)

        if id_ is not None:
            fields["metadata"] = MetadataField({"id": id_})

        if relation is not None:
            fields["label"] = LabelField(relation)

        return Instance(fields)

    def _apply_masking_mode(self, tokens, head, tail, head_type, tail_type):
        if self._masking_mode == "NER":
            head_replacement = f"__{head_type}__"
            tail_replacement = f"__{tail_type}__"
        elif self._masking_mode == "Grammar":
            head_replacement = "__SUB__"
            tail_replacement = "__OBJ__"
        elif self._masking_mode == "NER+Grammar":
            head_replacement = f"__{head_type}_SUB__"
            tail_replacement = f"__{tail_type}_OBJ__"
        elif self._masking_mode == "UNK":
            head_replacement = "__UNK__"
            tail_replacement = "__UNK__"
        elif self._masking_mode == "NER_NL":
            head_replacement = f"{head_type.lower()}"
            tail_replacement = f"{tail_type.lower()}"
        else:
            raise ValueError("Unknown masking mode '%s'" % self._masking_mode)

        tokens[head[0]: head[1] + 1] = [head_replacement] * (head[1] - head[0] + 1)
        tokens[tail[0]: tail[1] + 1] = [tail_replacement] * (tail[1] - tail[0] + 1)

        return tokens
