import torch
from torch import nn


class rnnNet(nn.Module):
    def __init__( self, in_dim, hidden_dim, n_layer, n_class ):
        super(rnnNet, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.n_layer,
            batch_first=True
        )
        self.classifier = nn.Linear(hidden_dim, n_class)

    def forward( self, x ):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 将输出的seq（序列）提取出来
        out = self.classifier(out)
        return out
