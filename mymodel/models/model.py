from torch.autograd import Function
import torch.nn as nn
import transformers as T
import torch
import torch.nn.init as init
from allennlp.data.tokenizers import WordTokenizer
from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.modules import InputVariationalDropout
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.util import replace_masked_values
from allennlp.modules.seq2seq_encoders import _Seq2SeqWrapper
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.token_indexers import SingleIdTokenIndexer


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class UnbiasedTopicalExtractorBERT(nn.Module):

    def __init__(self, max_seq_length, device):
        super(UnbiasedTopicalExtractorBERT, self).__init__()
        self.max_seq_length = max_seq_length
        self.device = device
        self.tokenizer = T.BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = T.BertModel.from_pretrained('bert-base-uncased')
        in_features = self.bert_model.config.hidden_size
        self._class_classification_layer = torch.nn.Linear(in_features, 2)
        self._group_classification_layer = torch.nn.Linear(in_features, 2)
        init.xavier_uniform_(self._class_classification_layer.weight)
        init.zeros_(self._class_classification_layer.bias)
        init.xavier_uniform_(self._group_classification_layer.weight)
        init.zeros_(self._group_classification_layer.bias)
        self._class_loss = torch.nn.CrossEntropyLoss()
        self._group_loss = torch.nn.CrossEntropyLoss()

    def forward(self, alpha, tweets, class_labels=None, group_labels=None):
        input_ids = []
        input_masks = []
        token_type_ids = []
        for tweet in tweets:
            # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for bert model.
            input_id = self.tokenizer.encode(tweet, add_special_tokens=False)
            # Account for [CLS] and [SEP] with "- 2"
            if len(input_id) > self.max_seq_length - 2:
                input_id = input_id[:(self.max_seq_length - 2)]

            input_id = [101] + input_id + [102]
            token_type_id = [0] * len(input_id)

            # The mask has 1 for real tokens and 0 for padding tokens.
            input_mask = [1] * len(input_id)

            # Zero-pad up to the sequence length.
            padding = [0] * (self.max_seq_length - len(input_id))
            input_id += padding
            input_mask += padding
            token_type_id += padding

            assert len(input_id) == self.max_seq_length
            assert len(input_mask) == self.max_seq_length
            assert len(token_type_id) == self.max_seq_length

            input_ids.append(input_id)
            input_masks.append(input_mask)
            token_type_ids.append(token_type_id)

        # Use the last layer hidden-state of the first token of the sequence ([CLS]) further processed by a Linear
        # layer and a Tanh activation function.
        feature = self.bert_model(input_ids=torch.tensor(input_ids).to(self.device),
                                  attention_mask=torch.tensor(input_masks).to(self.device),
                                  token_type_ids=torch.tensor(token_type_ids).to(self.device))[1]

        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_logits = self._class_classification_layer(feature)
        group_logits = self._group_classification_layer(reverse_feature)
        class_probs = torch.nn.functional.softmax(class_logits, dim=-1)
        group_probs = torch.nn.functional.softmax(group_logits, dim=-1)
        output_dict = {"class_logits": class_logits, "group_logits": group_logits,
                       "class_probs": class_probs, "group_probs": group_probs}
        if class_labels is not None:
            class_loss = self._class_loss(class_logits, class_labels.long().view(-1))
            output_dict["class_loss"] = class_loss
        if group_labels is not None:
            group_loss = self._group_loss(group_logits, group_labels.long().view(-1))
            output_dict["group_loss"] = group_loss

        return output_dict


class UnbiasedTopicalExtractorELMo(nn.Module):

    def __init__(self, device):
        super(UnbiasedTopicalExtractorELMo, self).__init__()
        self.device = device
        self.tokenizer = WordTokenizer()
        options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
        self.elmo_model = Elmo(options_file, weight_file, 1, do_layer_norm=False, dropout=0.5)
        self.rnn_input_dropout = InputVariationalDropout(0.0)
        self.encoder = _Seq2SeqWrapper(torch.nn.LSTM)(
            input_size=1024,
            hidden_size=300,
            num_layers=1,
            bidirectional=True)

        self._feature_feedforward_layer = torch.nn.Linear(600, 300)
        self._feature_feedforward_dropout = torch.nn.Dropout(0.5)
        self._feature_feedforward_activation = torch.nn.ReLU()
        self._class_classification_layer = torch.nn.Linear(300, 2)
        self._group_classification_layer = torch.nn.Linear(300, 2)
        init.xavier_uniform_(self._feature_feedforward_layer.weight)
        init.zeros_(self._feature_feedforward_layer.bias)
        init.xavier_uniform_(self._class_classification_layer.weight)
        init.zeros_(self._class_classification_layer.bias)
        init.xavier_uniform_(self._group_classification_layer.weight)
        init.zeros_(self._group_classification_layer.bias)
        self._class_loss = torch.nn.CrossEntropyLoss()
        self._group_loss = torch.nn.CrossEntropyLoss()

    def forward(self, alpha, tweets, class_labels=None, group_labels=None):
        tweets_tokens = []
        for tweet in tweets:
            tokens = self.tokenizer.tokenize(tweet)
            tweets_tokens.append([t.text for t in tokens])

        elmo_output = self.elmo_model(batch_to_ids(tweets_tokens).to(self.device))
        embedded_tweet = elmo_output["elmo_representations"][0]
        mask = elmo_output["mask"].float()
        embedded_tweet = self.rnn_input_dropout(embedded_tweet)
        # encode tweet, (batch_size, tweet_length, hidden_dim)
        encoded_tweet = self.encoder(embedded_tweet, mask)
        # The pooling layer -- max pooling.
        # (batch_size, model_dim)
        encode_max, _ = replace_masked_values(encoded_tweet, mask.unsqueeze(-1), -1e7).max(dim=1)
        feature = self._feature_feedforward_dropout(self._feature_feedforward_activation(self._feature_feedforward_layer(encode_max)))

        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_logits = self._class_classification_layer(feature)
        group_logits = self._group_classification_layer(reverse_feature)
        class_probs = torch.nn.functional.softmax(class_logits, dim=-1)
        group_probs = torch.nn.functional.softmax(group_logits, dim=-1)
        output_dict = {"class_logits": class_logits, "group_logits": group_logits,
                       "class_probs": class_probs, "group_probs": group_probs}
        if class_labels is not None:
            class_loss = self._class_loss(class_logits, class_labels.long().view(-1))
            output_dict["class_loss"] = class_loss
        if group_labels is not None:
            group_loss = self._group_loss(group_logits, group_labels.long().view(-1))
            output_dict["group_loss"] = group_loss

        return output_dict


class UnbiasedTopicalExtractorGloVe(nn.Module):

    def __init__(self, device, vocab_dir):
        super(UnbiasedTopicalExtractorGloVe, self).__init__()
        self.device = device
        vocab = Vocabulary()
        self.vocab = vocab.from_files(vocab_dir)
        self.embedding = Embedding(embedding_dim=300, trainable=True, num_embeddings=self.vocab.get_vocab_size("tokens"),
                                   pretrained_file="https://allennlp.s3.amazonaws.com/datasets/glove/glove.840B.300d.txt.gz")
        self.tokenizer = WordTokenizer()
        self.token_indexer = SingleIdTokenIndexer(lowercase_tokens=True)
        self.rnn_input_dropout = InputVariationalDropout(0.5)
        self.encoder = _Seq2SeqWrapper(torch.nn.LSTM)(
            input_size=300,
            hidden_size=300,
            num_layers=1,
            bidirectional=True)

        self._feature_feedforward_layer = torch.nn.Linear(600, 300)
        self._feature_feedforward_dropout = torch.nn.Dropout(0.5)
        self._feature_feedforward_activation = torch.nn.ReLU()
        self._class_classification_layer = torch.nn.Linear(300, 2)
        self._group_classification_layer = torch.nn.Linear(300, 2)
        init.xavier_uniform_(self._feature_feedforward_layer.weight)
        init.zeros_(self._feature_feedforward_layer.bias)
        init.xavier_uniform_(self._class_classification_layer.weight)
        init.zeros_(self._class_classification_layer.bias)
        init.xavier_uniform_(self._group_classification_layer.weight)
        init.zeros_(self._group_classification_layer.bias)
        self._class_loss = torch.nn.CrossEntropyLoss()
        self._group_loss = torch.nn.CrossEntropyLoss()

    def forward(self, alpha, tweets, class_labels=None, group_labels=None):
        batch_indices = []
        batch_masks = []
        max_len = 0
        for tweet in tweets:
            tokens = self.tokenizer.tokenize(tweet)
            indices = self.token_indexer.tokens_to_indices(tokens, self.vocab, index_name="tokens")["tokens"]
            max_len = max(len(indices), max_len)
            # The mask has 1 for real tokens and 0 for padding tokens.
            input_mask = [1] * len(indices)
            batch_indices.append(indices)
            batch_masks.append(input_mask)
        for indices, input_mask in zip(batch_indices, batch_masks):
            # Zero-pad up to the max sequence length within the batch.
            padding = [0] * (max_len - len(indices))
            indices += padding
            input_mask += padding
            assert len(indices) == max_len
            assert len(input_mask) == max_len

        batch_indices = torch.tensor(batch_indices).to(self.device)
        mask = torch.tensor(batch_masks).float().to(self.device)
        embedded_tweet = self.embedding(batch_indices)
        embedded_tweet = self.rnn_input_dropout(embedded_tweet)
        # encode tweet, (batch_size, tweet_length, hidden_dim)
        encoded_tweet = self.encoder(embedded_tweet, mask)
        # The pooling layer -- max pooling.
        # (batch_size, model_dim)
        encode_max, _ = replace_masked_values(encoded_tweet, mask.unsqueeze(-1), -1e7).max(dim=1)
        feature = self._feature_feedforward_dropout(self._feature_feedforward_activation(self._feature_feedforward_layer(encode_max)))

        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_logits = self._class_classification_layer(feature)
        group_logits = self._group_classification_layer(reverse_feature)
        class_probs = torch.nn.functional.softmax(class_logits, dim=-1)
        group_probs = torch.nn.functional.softmax(group_logits, dim=-1)
        output_dict = {"class_logits": class_logits, "group_logits": group_logits,
                       "class_probs": class_probs, "group_probs": group_probs}
        if class_labels is not None:
            class_loss = self._class_loss(class_logits, class_labels.long().view(-1))
            output_dict["class_loss"] = class_loss
        if group_labels is not None:
            group_loss = self._group_loss(group_logits, group_labels.long().view(-1))
            output_dict["group_loss"] = group_loss

        return output_dict
