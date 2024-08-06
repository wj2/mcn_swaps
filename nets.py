
import torch.nn as nn


class SimpleRecurrent(nn.Module):
    def __init__(self, inp_size, num_h, out_size, net_type=nn.RNN):
        super(SimpleRecurrent, self).__init__()
        self.hidden_dim = num_h
        self.out_dim = out_size
        self.in_dim = inp_size
        self.recurrent_net = net_type(inp_size, num_h)
        self.linear = nn.Linear(num_h, out_size)

    def forward(self, x):
        out, hidden = self.recurrent_net(x)
        x = self.linear(out)
        return x


class SimpleRNN(SimpleRecurrent):
    def __init__(self, *args, **kwargs):
        super(SimpleRNN, self).__init__(*args, **kwargs, net_type=nn.RNN)
        

class SimpleLSTM(SimpleRecurrent):
    def __init__(self, *args, **kwargs):
        super(SimpleLSTM, self).__init__(*args, **kwargs, net_type=nn.LSTM)
