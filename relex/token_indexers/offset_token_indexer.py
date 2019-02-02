import logging
from typing import Dict, List
import itertools

from overrides import overrides

from allennlp.common.util import pad_sequence_to_length
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@TokenIndexer.register("offset")
class OffsetTokenIndexer(TokenIndexer[int]):
    """
    This :class:`TokenIndexer` represents token offsets as single integers.
    Parameters
    ----------
    """

    # pylint: disable=no-self-use
    def __init__(self, token_attribute: str = "offset") -> None:
        self._token_attribute = token_attribute

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        pass

    @overrides
    def tokens_to_indices(
        self, tokens: List[Token], vocabulary: Vocabulary, index_name: str
    ) -> Dict[str, List[int]]:
        indices: List[int] = []

        for token in tokens:
            offset = getattr(token, self._token_attribute, None)
            if offset is not None:
                indices.append(offset)
            else:
                logger.warning("Token had no offset attribute: %s", token.text)

        return {index_name: indices}

    @overrides
    def get_padding_token(self) -> int:
        return -1

    @overrides
    def get_padding_lengths(
        self, token: int
    ) -> Dict[str, int]:  # pylint: disable=unused-argument
        return {}

    @overrides
    def pad_token_sequence(
        self,
        tokens: Dict[str, List[int]],
        desired_num_tokens: Dict[str, int],
        padding_lengths: Dict[str, int],
    ) -> Dict[str, List[int]]:  # pylint: disable=unused-argument
        return {
            key: pad_sequence_to_length(val, desired_num_tokens[key])
            for key, val in tokens.items()
        }
