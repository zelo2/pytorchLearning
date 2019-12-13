import torch
from torch import nn

lstm = nn.LSTM(
    input_size=20,
    hidden_size=50,
    num_layers=10,
)
print(lstm.weight_ih_l0)
print(lstm.weight_ih_l0.size())
print(lstm.bias_ih_l0.size())
