from torch.utils.data import DataLoader
from torch import nn
from torchtext.vocab import build_vocab_from_iterator
from sklearn.model_selection import train_test_split
from torchtext.data.utils import get_tokenizer
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
import pandas as pd
import torch
import re
from tqdm import tqdm
import time

from model import  MTCNNLSTM
from utils import preprocess_text
from loss import Loss
from train import TrainVal


def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_text, _label) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)


if __name__ == '__main__':
    with open('data/apparel/positive.review', encoding='utf8', errors='ignore') as f:
        df_pos = pd.read_xml(f)
    with open('data/apparel/negative.review', encoding='utf8', errors='ignore') as f:
        df_neg = pd.read_xml(f)

    df_pos['label'] = 0
    df_neg['label'] = 1

    x = df_pos['review_text'].tolist() + df_neg['review_text'].tolist()
    y = df_pos['label'].tolist() + df_neg['label'].tolist()
    x_clean = []
    for sentence in tqdm(x):
        x_clean.append(preprocess_text(sentence))
    train_iter = list(zip(x_clean, y))

    tokenizer = get_tokenizer('basic_english')


    def yield_tokens(data_iter):
        for text, _ in data_iter:
            yield tokenizer(text)


    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: int(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = len(set([label for (text, label) in train_iter]))
    vocab_size = len(vocab)
    emsize = 256
    model = MTCNNLSTM(vocab_size, emsize).to(device)

    # Hyperparameters
    EPOCHS = 100  # epoch
    LR = 0.1  # learning rate
    BATCH_SIZE = 64  # batch size for training

    # criterion = torch.nn.CrossEntropyLoss()
    train_dataset = to_map_style_dataset(train_iter)
    num_train = int(len(train_dataset) * 0.95)
    split_train_, split_valid_ = \
        random_split(train_dataset, [num_train, len(train_dataset) - num_train])

    train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE,
                                  shuffle=True, collate_fn=collate_batch)
    valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE,
                                  shuffle=True, collate_fn=collate_batch)

    training = TrainVal(model, train_dataloader, valid_dataloader)
    training.start_training()