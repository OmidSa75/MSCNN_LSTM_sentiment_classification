import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from loss import Loss

RED = "\033[1;31m"
BLUE = "\033[1;34m"
CYAN = "\033[1;36m"
GREEN = "\033[0;32m"
RESET = "\033[0;0m"
BOLD = "\033[;1m"
REVERSE = "\033[;7m"


class TrainVal:
    def __init__(self, model: nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.criterion = Loss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.1)

        self.total_acc = None

    def train(self):
        self.model.train()
        total_acc, total_count = 0.0, 0
        total_loss = 0.0
        f1score = 0.0

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
            f1score += f1_score(label.data.cpu().numpy(), predited_label.argmax(1).data.cpu().numpy())

        print(BLUE, 'Train | accuracy {:8.3f} | F1 Score: {:.4f} | loss {:.5f}'.format(total_acc / total_count,
                                                                                       total_loss / len(
                                                                                           self.train_dataloader),
                                                                                       f1score / len(
                                                                                           self.train_dataloader)),
              RESET)

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_acc, total_count = 0.0, 0
        total_loss = 0.0
        f1score = 0.0
        for idx, (label, text, offsets) in enumerate(self.val_dataloader):
            predited_label, private_features, shared_features = self.model(text, offsets)
            loss = self.criterion(predited_label, label, shared_features, private_features)
            total_acc += (predited_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
            total_loss += loss.data
            f1score += f1_score(label.cpu().numpy(), predited_label.argmax(1).cpu().numpy())

        print(GREEN, 'Test | accuracy {:8.3f} | F1 Score: {:.4f} | loss {:.5f}'.format(total_acc / total_count,
                                                                                       total_loss / len(self.val_dataloader),
                                                                                       f1score / len(self.val_dataloader)),
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
