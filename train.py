import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from loss import Loss

RED = "\033[1;31m"
BLUE = "\033[1;34m"
CYAN = "\033[1;36m"
GREEN = "\033[0;32m"
RESET = "\033[0;0m"
BOLD = "\033[;1m"
REVERSE = "\033[;7m"


class TrainVal:
    def __init__(self, model: nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader, criterion):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.criterion = criterion
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.scheduler = scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.1)

        self.total_acc = None

    def train(self):
        self.model.train()
        total_acc, total_count = 0.0, 0
        total_loss = 0.0

        for idx, (label, text, offsets) in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()
            predited_label, private_features, shared_features = self.model(text, offsets)
            loss = self.criterion(predited_label, label, shared_features, private_features)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            self.optimizer.step()
            total_acc += (predited_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
            total_loss += loss.data

        print(BLUE, 'Train | accuracy {:8.3f} | loss {:.5f}'.format(total_acc / total_count, total_loss / total_count),
              RESET)

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_acc, total_count = 0.0, 0
        total_loss = 0.0
        for idx, (label, text, offsets) in enumerate(self.val_dataloader):
            predited_label, private_features, shared_features = self.model(text, offsets)
            loss = self.criterion(predited_label, label, shared_features, private_features)
            total_acc += (predited_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
            total_loss += loss.data
        print(GREEN, 'Test | accuracy {:8.3f} | loss {:.5f}'.format(total_acc / total_count, total_loss / total_count),
              RESET)
        return total_acc

    def start_training(self, epochs=20):

        for epoch in range(1, epochs, 1):
            print('-' * 59)
            self.train()
            val_acc = self.evaluate()

            if self.total_acc is not None and self.total_acc > val_acc:
                self.scheduler.step()
            else:
                self.total_acc = val_acc

            print("End of epoch : {}".format(epoch))
            print('-' * 59)