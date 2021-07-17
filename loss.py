import torch
from torch import nn


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.gamma = 0.0001

    def forward(self, predict, label, shared_features: torch.Tensor, private_features: torch.Tensor):
        cls_loss = self.cross_entropy(predict, label)
        # orthogonal_loss = torch.sum(torch.matmul(shared_features.T, private_features)**2)
        orthogonal_loss = torch.sum(torch.inner(shared_features, private_features)**2)
        return cls_loss + orthogonal_loss * self.gamma
