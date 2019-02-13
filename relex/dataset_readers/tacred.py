from typing import Dict, List, Tuple, Optional
import json
import logging

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, SpanField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("tacred")
class TacredDatasetReader(DatasetReader):
    """
    Reads a JSON file containing examples from the TACRED dataset,
    and creates a dataset suitable for relation classification.
    The JSON could have other fields, too, but they are ignored.
    The output of ``read`` is a list of ``Instance`` s with the fields:
        text: ``TextField``
        head: ``SpanField``
        tail: ``SpanField``
        label: ``LabelField``
    Parameters
    ----------
    max_len : ``int``
        Limit the number of tokens for each text. This is important for computing the relative offset
        of head and tail entities. (TODO: maybe there's a better way to handle this)
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
    """

    def __init__(
        self,
        max_len: int,
        masking_mode: str = None,
        lazy: bool = False,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
    ) -> None:
        super().__init__(lazy)
        self._max_len = max_len
        self._tokenizer = tokenizer or WordTokenizer(
            word_splitter=JustSpacesWordSplitter()
        )
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._masking_mode = masking_mode

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r") as data_file:
            logger.info("Reading TACRED instances from json dataset at: %s", file_path)
            data = json.load(data_file)
            for example in data:
                tokens = example["token"]
                relation = example["relation"]

                id_ = example["id"]

                head = (example["subj_start"], example["subj_end"])
                tail = (example["obj_start"], example["obj_end"])

                if self._masking_mode is not None:
                    ner_labels = example['stanford_ner']
                    tokens = self._apply_masking_mode(tokens, head, tail, ner_labels)

                text = " ".join(tokens)

                yield self.text_to_instance(text, head, tail, id_, relation)

    @overrides
    def text_to_instance(
        self,
        text: str,
        head: Tuple[int, int],
        tail: Tuple[int, int],
        id_: Optional[str] = None,
        relation: Optional[str] = None,
    ) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ

        tokenized_text = self._tokenizer.tokenize(text)
        tokenized_text = tokenized_text[: self._max_len]

        head_start, head_end = head
        tail_start, tail_end = tail

        text_tokens_field = TextField(tokenized_text, self._token_indexers)
        # SpanField expects an inclusive end index
        fields = {
            "text": text_tokens_field,
            "head": SpanField(head_start, head_end, sequence_field=text_tokens_field),
            "tail": SpanField(tail_start, tail_end, sequence_field=text_tokens_field),
        }

        if id_ is not None:
            fields["metadata"] = MetadataField({"id": id_})

        if relation is not None:
            fields["label"] = LabelField(relation)

        return Instance(fields)

    def _apply_masking_mode(self, tokens, head, tail, ner_labels):
        if self._masking_mode == 'NER':
            head_replacement = f'__{ner_labels[head[0]]}__'
            tail_replacement = f'__{ner_labels[tail[0]]}__'
        elif self._masking_mode == 'Grammar':
            head_replacement = '__SUB__'
            tail_replacement = '__OBJ__'
        elif self._masking_mode == 'NER+Grammar':
            head_replacement = f'__{ner_labels[head[0]]}_SUB__'
            tail_replacement = f'__{ner_labels[tail[0]]}_OBJ__'
        elif self._masking_mode == 'UNK':
            head_replacement = '__UNK__'
            tail_replacement = '__UNK__'
        else:
            raise RuntimeError('Unknown masking mode provided')

        tokens[head[0]:head[1]+1] = [head_replacement] * (head[1]-head[0]+1)
        tokens[tail[0]:tail[1]+1] = [tail_replacement] * (tail[1]-tail[0]+1)

        return tokens
