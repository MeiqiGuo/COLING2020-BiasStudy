"""
This script predicts the relevance score to a specific topic on all tweets with a well-trained model.
"""

import os

from typing import Iterable, Dict
import torch
import argparse
import csv

torch.manual_seed(1)

from allennlp.common.params import Params
from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data import Instance
from allennlp.data.iterators import BasicIterator
from allennlp.data.iterators import DataIterator
from tqdm import tqdm
from allennlp.nn import util as nn_util
from models import topical_extractor, topical_extractor_bert
from sklearn.metrics import precision_score, recall_score, f1_score


def tonp(tsr): return tsr.detach().cpu().numpy()


class Predictor:
    def __init__(self, model: Model, iterator: DataIterator,
                 cuda_device: int = -1) -> None:
        self.model = model
        self.iterator = iterator
        self.vocab = model.vocab
        self.cuda_device = cuda_device

    def predict(self, ds: Iterable[Instance]) -> Dict:
        pred_generator = self.iterator(ds, num_epochs=1, shuffle=False)
        self.model.eval()
        pred_generator_tqdm = tqdm(pred_generator,
                                   total=self.iterator.get_num_batches(ds))
        output = {"prediction": []}
        with torch.no_grad():
            for batch in pred_generator_tqdm:
                batch = nn_util.move_to_device(batch, self.cuda_device)
                out_dict = self.model(**batch)
                output["prediction"] += [probs[self.vocab._token_to_index["labels"]["1"]]
                                         for probs in tonp(out_dict["label_probs"])]
        return output


def write_results(input_path, predict_output, output_path):
    data = []
    labels = {"lib":[], "con":[]}
    preds = {"lib":[], "con":[]}

    with open(input_path, 'r') as tweet_file:
        csv_reader = csv.reader(tweet_file, delimiter=',', quoting=csv.QUOTE_ALL, escapechar='\\')
        for row in csv_reader:
            if len(row) == 3 and row[2] in ["0", "1"]:
                data.append(row)
            else:
            # These were cases where the row has wrong columns; we'll just skip them.
                continue

    predictions = predict_output["prediction"]
    assert len(predictions) == len(data), "Eval set and prediction results don't match: {0} - {1}"\
        .format(len(predictions), len(data))

    with open(output_path, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',', escapechar='\\')
        for sample, score in zip(data, predictions):
            writer.writerow(sample + [score])
            group = sample[0]
            labels[group].append(int(sample[2]))
            preds[group].append(int(score >= 0.5))

    print("Precision score: {}".format(precision_score(labels["con"] + labels["lib"], preds["con"] + preds["lib"])))
    print("Recall score: {}".format(recall_score(labels["con"] + labels["lib"], preds["con"] + preds["lib"])))
    print("F1 score: {}".format(f1_score(labels["con"] + labels["lib"], preds["con"] + preds["lib"])))
    print("Precision score for con group: {}".format(precision_score(labels["con"], preds["con"])))
    print("Recall score for con group: {}".format(recall_score(labels["con"], preds["con"])))
    print("F1 score for con group: {}".format(f1_score(labels["con"], preds["con"])))
    print("Precision score for lib group: {}".format(precision_score(labels["lib"], preds["lib"])))
    print("Recall score for lib group: {}".format(recall_score(labels["lib"], preds["lib"])))
    print("F1 score for lib group: {}".format(f1_score(labels["lib"], preds["lib"])))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir_path", help="The path of the model directory")
    parser.add_argument("--pred_file", help="The path of the file to be predicted", default=None)
    args = parser.parse_args()
    #USE_GPU = torch.cuda.is_available()
    USE_GPU = False
    params = Params.from_file(os.path.join(args.dir_path, 'config.json'))
    model_params = params.get("model")
    vocab = Vocabulary.from_files(os.path.join(args.dir_path, "vocabulary"))
    print(vocab._index_to_token["labels"])

    model = Model.from_params(model_params, vocab=vocab)
    with open(os.path.join(args.dir_path, "weights.th"), 'rb') as f:
        model.load_state_dict(torch.load(f))

    # dataset_reader
    dataset_reader_params = params.get("dataset_reader")
    dataset_reader = DatasetReader.from_params(dataset_reader_params)

    test_dataset = dataset_reader.read(args.pred_file)
    print(len(test_dataset))

    # iterate over the dataset without changing its order
    seq_iterator = BasicIterator(batch_size=512)
    seq_iterator.index_with(vocab)

    predictor = Predictor(model, seq_iterator, cuda_device=0 if USE_GPU else -1)
    test_preds = predictor.predict(test_dataset)
    output_path = os.path.join(args.dir_path, args.pred_file.split("/")[-1].split(".")[0] + "_prediction.csv")
    write_results(args.pred_file, test_preds, output_path)

