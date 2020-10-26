from allennlp.data.tokenizers import WordTokenizer
from collections import defaultdict


class Vocab(object):
    def __init__(self):
        self.tweets = []
        self.vocab = defaultdict(int)
        self.tokenizer = WordTokenizer()

    def build_vocab(self, tweets):
        self.tweets = tweets
        for tweet in self.tweets:
            for t in self.tokenizer.tokenize(tweet):
                self.vocab[t.text] += 1
        self.vocab = sorted(self.vocab.items(), key=lambda item: item[1], reverse=True)
        print(len(self.vocab))
        self.vocab = [x for x, f in self.vocab if f > 1]
        print(len(self.vocab))

    def write_vocab(self, vocab_path):
        pass
    def load_vocab(self, vocab_path):
        pass
