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


def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predited_label, private_features, shared_features = model(text, offsets)
        loss = criterion(predited_label, label, shared_features, private_features)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predited_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc / total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()


@torch.no_grad()
def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predited_label, _, _ = model(text, offsets)
            # loss = criterion(predited_label, label)
            total_acc += (predited_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count


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
    EPOCHS = 10  # epoch
    LR = 5  # learning rate
    BATCH_SIZE = 64  # batch size for training

    # criterion = torch.nn.CrossEntropyLoss()
    criterion = Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
    total_accu = None
    train_dataset = to_map_style_dataset(train_iter)
    num_train = int(len(train_dataset) * 0.95)
    split_train_, split_valid_ = \
        random_split(train_dataset, [num_train, len(train_dataset) - num_train])

    train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE,
                                  shuffle=True, collate_fn=collate_batch)
    valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE,
                                  shuffle=True, collate_fn=collate_batch)

    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        train(train_dataloader)
        accu_val = evaluate(valid_dataloader)
        if total_accu is not None and total_accu > accu_val:
            scheduler.step()
        else:
            total_accu = accu_val
        print('-' * 59)
        print('| end of epoch {:3d} | time: {:5.2f}s | '
              'valid accuracy {:8.3f} '.format(epoch,
                                               time.time() - epoch_start_time,
                                               accu_val))
        print('-' * 59)