import torch
from torch import nn
from torch.nn import functional as F


class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(512 * 759, 256)
        self.tanh = nn.Tanh()

    def init_weights(self):
        initrange = 0.5
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(self.dropout(x))
        out = self.tanh(x)
        return out


class LocalEncoder(nn.Module):
    def __init__(self):
        super(LocalEncoder, self).__init__()
        self.cnn3 = nn.Conv1d(1, 512, kernel_size=3)
        self.cnn4 = nn.Conv1d(1, 512, kernel_size=4)
        self.cnn5 = nn.Conv1d(1, 512, kernel_size=5)
        self.pooling = nn.MaxPool1d(2, 2)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.cnn3.weight.data.uniform_(-initrange, initrange)
        self.cnn4.weight.data.uniform_(-initrange, initrange)
        self.cnn5.weight.data.uniform_(-initrange, initrange)
        self.cnn3.bias.data.zero_()
        self.cnn4.bias.data.zero_()
        self.cnn5.bias.data.zero_()

    def forward(self, vectors: torch.Tensor):
        """

        :param vectors: (batch, channel, word_dims) tensors.
        :return:
        """
        vectors = vectors.unsqueeze(1)
        x3 = F.relu(self.cnn3(vectors))
        x4 = F.relu(self.cnn4(vectors))
        x5 = F.relu(self.cnn5(vectors))
        x = torch.cat((x3, x4, x5), dim=2)

        return x


class GlobalEncoder(nn.Module):
    def __init__(self):
        super(GlobalEncoder, self).__init__()
        self.lstm = nn.LSTM(256, hidden_size=256, num_layers=2, batch_first=True)

    def forward(self, x):
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        return lstm_out


class WordEmbed(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(WordEmbed, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, text, offsets):
        return self.embedding(text, offsets)


class MTCNNLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(MTCNNLSTM, self).__init__()
        self.private = LocalEncoder()
        self.share = GlobalEncoder()
        self.embedding = WordEmbed(vocab_size, embed_dim)
        self.fusion_net = FusionNet()

        self.fc = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 2),
        )

    def forward(self, text, offset):
        embed = self.embedding(text, offset)
        shared_features = self.share(embed).squeeze(1)
        private_features = self.private(embed)
        private_features = self.fusion_net(private_features)

        cat_features = torch.cat((private_features, shared_features), dim=1)

        cls = self.fc(cat_features)
        return cls, private_features, shared_features
