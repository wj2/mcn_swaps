
import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self, inp_size, num_h, out_size):
        super(SimpleNet, self).__init__()
        self.hidden_dim = num_h
        self.out_dim = out_size
        self.in_dim = inp_size
        self.lstm = nn.LSTM(inp_size, num_h)
        self.linear = nn.Linear(num_h, out_size)

    def forward(self, x):
        out, hidden = self.lstm(x)
        x = self.linear(out)
        return x
