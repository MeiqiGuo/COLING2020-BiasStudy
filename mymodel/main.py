import random
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import copy
import os
from tqdm import tqdm
import argparse
from allennlp.data.vocabulary import Vocabulary
from dataset.data_loader import TweetReader, TweetInstanceReader
from models.model import UnbiasedTopicalExtractorBERT, UnbiasedTopicalExtractorELMo, UnbiasedTopicalExtractorGloVe


parser = argparse.ArgumentParser()
parser.add_argument("--base_model", help="The name of the base model", type=str, default=None, choices=["bert", "elmo", "glove"])
parser.add_argument("--train_data_path", help="The path of the training dataset", type=str, default=None)
parser.add_argument("--val_data_path", help="The path of the val dataset", type=str, default=None)
parser.add_argument("--test_data_path", help="The path of the test dataset", type=str, default=None)
parser.add_argument("--save_root", help="The root of saved files", type=str, default=None)
parser.add_argument('--lr', help='learning rate', default=3e-5, type=float)
parser.add_argument('--batch_size', help='batch size', default=32, type=int)
parser.add_argument('--test_batch_size', help='batch size', default=256, type=int)
parser.add_argument('--n_epoch', help='number of epochs', default=5, type=int)
parser.add_argument('--gamma', help='gamma for computing alpha', default=10, type=int)
parser.add_argument('--max_seq_length', help='maximum length of input tokens', default=128, type=int)
parser.add_argument('--train', help='train model', action='store_true')
parser.add_argument('--test', dest='test', help='test model', action='store_true')

args = parser.parse_args()
print(args)

# Reproducibility
np.random.seed(123)
random.seed(456)
torch.manual_seed(789)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set model
if args.base_model == "bert":
    model = UnbiasedTopicalExtractorBERT(args.max_seq_length, device)
elif args.base_model == "elmo":
    model = UnbiasedTopicalExtractorELMo(device)
elif args.base_model == "glove":
    vocab_dir = os.path.join(args.save_root, 'vocab')
    if os.path.exists(vocab_dir):
        # Load vocabulary
        model = UnbiasedTopicalExtractorGloVe(device, vocab_dir)
    else:
        # Build vocabulary
        reader = TweetInstanceReader()
        train_dataset = reader.read(args.train_data_path)
        val_dataset = reader.read(args.val_data_path)
        vocab = Vocabulary.from_instances(train_dataset + val_dataset)
        #print("vocab index 0 is {}".format(vocab.get_token_from_index(0)))
        #print("oov index is {}".format(vocab._token_to_index["tokens"][vocab._oov_token]))
        vocab.save_to_files(vocab_dir)
        model = UnbiasedTopicalExtractorGloVe(device, vocab_dir)
else:
    pass

model = model.to(device)

# Training
if args.train:
    # Check save root
    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root)

    # Load data
    dataset = {}
    data_loader = {}
    for phase, path in zip(["train", "val"], [args.train_data_path, args.val_data_path]):
        dataset[phase] = TweetReader(path)
        data_loader[phase] = DataLoader(dataset=dataset[phase], batch_size=args.batch_size, shuffle=True, num_workers=8)

    # Set loss
    loss_class = torch.nn.CrossEntropyLoss()
    loss_group = torch.nn.CrossEntropyLoss()

    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training
    best_class_auc = 0
    patient = 0
    for epoch in range(args.n_epoch):
        if patient >= 10:
            break
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            all_batch_group_loss = 0
            all_batch_class_loss = 0
            all_batch_group_corrects = 0
            all_batch_class_auc = 0
            len_dataloader = len(data_loader[phase])
            for batch_idx, (tweets, class_labels, group_labels) in enumerate(data_loader[phase]):
                # Set alpha
                if phase == "train":
                    p = float(batch_idx + epoch * len_dataloader) / (args.n_epoch * len_dataloader)
                    alpha = 2. / (1. + np.exp(-args.gamma * p)) - 1
                else:
                    alpha = 0

                class_labels = class_labels.to(device)
                group_labels = group_labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    output_dict = model(alpha, tweets, class_labels, group_labels)
                    losses = output_dict["class_loss"] + output_dict["group_loss"]
                    _, group_preds = torch.max(output_dict["group_probs"], 1)
                    class_preds = [probs[1].item() for probs in output_dict["class_probs"]]
                    try:
                        class_auc = roc_auc_score(class_labels.cpu(), class_preds)
                    except ValueError:
                        class_auc = 0.0
                    if phase == "train":
                        losses.backward()
                        optimizer.step()

                all_batch_class_loss += output_dict["class_loss"] * len(tweets)
                all_batch_group_loss += output_dict["group_loss"] * len(tweets)
                all_batch_group_corrects += torch.sum(group_preds == group_labels.data)
                all_batch_class_auc += class_auc * len(tweets)

            epoch_class_loss = all_batch_class_loss / dataset[phase].__len__()
            epoch_group_loss = all_batch_group_loss / dataset[phase].__len__()
            epoch_group_acc = float(all_batch_group_corrects) / dataset[phase].__len__()
            epoch_class_auc = all_batch_class_auc / dataset[phase].__len__()
            print("Epoch {0} ({1}): class loss -- {2}; group loss -- {3}; class AUC -- {4}; group accuracy -- {5}\n".
                  format(epoch, phase, epoch_class_loss, epoch_group_loss, epoch_class_auc, epoch_group_acc))
            if phase == 'val':
                if epoch_class_auc > best_class_auc:
                    best_class_auc = epoch_class_auc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, os.path.join(args.save_root, 'BEST.pth'))
                    patient = 0
                else:
                    patient += 1


# Testing
if args.test:
    dataset = TweetReader(args.test_data_path)
    data_loader = DataLoader(dataset=dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=8)
    model.load_state_dict(torch.load(os.path.join(args.save_root, 'BEST.pth')))
    model.eval()
    all_group_corrects = 0
    all_class_preds = []
    for tweets, class_labels, group_labels in tqdm(data_loader):
        with torch.set_grad_enabled(False):
            alpha = 0
            output_dict = model(alpha, tweets)
            _, group_preds = torch.max(output_dict["group_probs"], 1)
            class_preds = [probs[1].item() for probs in output_dict["class_probs"]]
            all_class_preds += class_preds
            all_group_corrects += torch.sum(group_preds.cpu() == group_labels.data).item()

    group_acc = float(all_group_corrects) / dataset.__len__()
    print("The accuracy for predicting group: {}\n".format(group_acc))
    output_path = os.path.join(args.save_root, args.test_data_path.split("/")[-1].split(".")[0] + "_prediction.csv")
    dataset.evaluate(all_class_preds, output_path)


