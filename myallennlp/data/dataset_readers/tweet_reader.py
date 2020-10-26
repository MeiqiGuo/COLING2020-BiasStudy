from typing import Dict
import csv
import logging
import emoji
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer, Token


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("tweet_reader")
class TweetReader(DatasetReader):
    """
    Reads a file containing tweets in both con and lib groups.
    This data is formatted as csv, one tweet instance per line.  Three columns in the data are
    "group", "tweet" and "label".
    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        See :class:`TokenIndexer`.
    """

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, 'r') as tweet_file:
            logger.info("Reading tweet instances from dataset at: %s", file_path)
            csv_reader = csv.reader(tweet_file, delimiter=',', quoting=csv.QUOTE_ALL, escapechar='\\')
            for row in csv_reader:
                if len(row) == 3:
                    _, tweet, label = row
                else:
                    # These were cases where the row has wrong columns; we'll just skip them.
                    continue
                if label not in ['0', '1']:
                # These were cases where the annotators disagreed; we'll just skip them.
                    continue
                yield self.text_to_instance(tweet, label)

    @overrides
    def text_to_instance(self,  # type: ignore
                         tweet: str,
                         label: str = None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        tokens = self._tokenizer.tokenize(tweet)
        fields['tokens'] = TextField(tokens, self._token_indexers)
        if label:
            fields['label'] = LabelField(label)
        return Instance(fields)



