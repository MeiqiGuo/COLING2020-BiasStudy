import torch.utils.data as data
import csv
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from typing import Dict
from allennlp.data.instance import Instance
from overrides import overrides
from allennlp.data.fields import Field, TextField, LabelField, MetadataField


class TweetReader(data.Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, 'r') as tweet_file:
            csv_reader = csv.reader(tweet_file, delimiter=',', quoting=csv.QUOTE_ALL, escapechar='\\')
            for row in csv_reader:
                if len(row) == 3:
                    group, tweet, label = row
                else:
                    # These were cases where the row has wrong columns; we'll just skip them.
                    continue
                if label not in ['0', '1']:
                    continue
                self.data.append(row)
        print("Reading {0} tweet instances from dataset at: {1}".format(len(self.data), file_path))

    def __getitem__(self, item):
        group, tweet, class_label = self.data[item]
        if group == "lib":
            group_label = 1
        elif group == "con":
            group_label = 0
        else:
            print("The group label is neither lib nor con: {}".format(group))
            assert False
        return tweet, int(class_label), group_label

    def __len__(self):
        return len(self.data)

    def evaluate(self, pred_scores, output_path):
        assert len(pred_scores) == len(self.data), "Eval set and prediction results don't match: {0} - {1}" \
            .format(len(pred_scores), len(self.data))
        labels = {"lib": [], "con": []}
        preds = {"lib": [], "con": []}
        scores = {"lib": [], "con": []}
        with open(output_path, "w") as csv_file:
            writer = csv.writer(csv_file, delimiter=',', escapechar='\\')
            for sample, score in zip(self.data, pred_scores):
                writer.writerow(sample + [score])
                group = sample[0]
                labels[group].append(int(sample[2]))
                preds[group].append(int(score >= 0.5))
                scores[group].append(score)

        print("Precision score: {}".format(precision_score(labels["con"] + labels["lib"], preds["con"] + preds["lib"])))
        print("Recall score: {}".format(recall_score(labels["con"] + labels["lib"], preds["con"] + preds["lib"])))
        print("F1 score: {}".format(f1_score(labels["con"] + labels["lib"], preds["con"] + preds["lib"])))
        print("AUC score: {}".format(roc_auc_score(labels["con"] + labels["lib"], scores["con"] + scores["lib"])))
        print("Precision score for con group: {}".format(precision_score(labels["con"], preds["con"])))
        print("Recall score for con group: {}".format(recall_score(labels["con"], preds["con"])))
        print("F1 score for con group: {}".format(f1_score(labels["con"], preds["con"])))
        print("AUC score for con group: {}".format(roc_auc_score(labels["con"], scores["con"])))
        print("Precision score for lib group: {}".format(precision_score(labels["lib"], preds["lib"])))
        print("Recall score for lib group: {}".format(recall_score(labels["lib"], preds["lib"])))
        print("F1 score for lib group: {}".format(f1_score(labels["lib"], preds["lib"])))
        print("AUC score for lib group: {}".format(roc_auc_score(labels["lib"], scores["lib"])))
        return


class TweetInstanceReader(DatasetReader):
    """
    Reads a file containing tweets in both con and lib groups.
    This data is formatted as csv, one tweet instance per line.  Three columns in the data are
    "group", "tweet" and "label".
    """

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer(lowercase_tokens=True)}

    @overrides
    def _read(self, file_path: str):
        with open(file_path, 'r') as tweet_file:
            print("Reading tweet instances from dataset at: %s", file_path)
            csv_reader = csv.reader(tweet_file, delimiter=',', quoting=csv.QUOTE_ALL, escapechar='\\')
            for row in csv_reader:
                if len(row) == 3:
                    group, tweet, label = row
                else:
                    # These were cases where the row has wrong columns; we'll just skip them.
                    continue
                if label not in ['0', '1']:
                # These were cases where the annotators disagreed; we'll just skip them.
                    continue
                yield self.text_to_instance(tweet, label, group)

    @overrides
    def text_to_instance(self,  # type: ignore
                         tweet: str,
                         label: str = None,
                         group: str = None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        tokens = self._tokenizer.tokenize(tweet)
        fields['tokens'] = TextField(tokens, self._token_indexers)
        if label:
            fields['class_label'] = LabelField(int(label), label_namespace="class_labels", skip_indexing=True)
        if group:
            fields['group_label'] = LabelField(group, label_namespace="group_labels")
        return Instance(fields)
