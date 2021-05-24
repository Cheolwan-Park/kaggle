import torch
from torch import Tensor, nn
from torch.nn.init import xavier_normal_
import torch.nn.functional as F
import numpy as np


# h_new = tanh(W_x*x + W_h*h + b)
# y = tanh(W_y*h_new + b)
class RNN(nn.Module):
    def __init__(self, hidden_size: int, emb_size: int):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.W_x = nn.Parameter(xavier_normal_(torch.empty(1, hidden_size, emb_size)),      # (1, H, D)
                                requires_grad=True).float()
        self.W_h = nn.Parameter(xavier_normal_(torch.empty(1, hidden_size, hidden_size)),   # (1, H, H)
                                requires_grad=True).float()
        self.W_y = nn.Parameter(xavier_normal_(torch.empty(1, emb_size, hidden_size)))      # (1, D, H)
        self.B_h = nn.Parameter(torch.from_numpy(np.zeros([1, hidden_size, 1])),            # (1, H, 1)
                                requires_grad=True).float()
        self.B_y = nn.Parameter(torch.from_numpy(np.zeros([1, emb_size, 1])),               # (1, D, 1)
                                requires_grad=True).float()

    def get_initial_h(self, batch_size: int):
        return torch.from_numpy(np.zeros([batch_size, self.hidden_size, 1])).float()

    def forward(self, x: Tensor, h: Tensor):        # x: (B, D, 1), h: (B, H, 1)
        wx = torch.matmul(self.W_x, x)              # (1, H, D) * (B, D, 1) = (B, H, 1)
        wh = torch.matmul(self.W_h, h)              # (1, H, H) * (B, H, 1) = (B, H, 1)
        hidden = torch.tanh(wx + wh + self.B_h)     # (B, H, 1)
        y = torch.matmul(self.W_y, hidden)          # (1, D, H) * (B, H, 1) = (B, D, 1)
        y = torch.tanh(y + self.B_y)                # (B, D, 1)
        return y, hidden


# https://wikidocs.net/60762 참조
# input: x(B, D, 1), h(B, H, 1), C(1, H, 1)
# i = sigmoid(W_xi*x + W_hi*h + b_i)        (1, H, 1)
# g = tanh(W_xg*x + W_hg*h + b_h)           (1, H, 1)
# f = sigmoid(W_xf*x + W_hf*h + b_f)        (1, H, 1)
# C_new = f x C + i x g                     (1, H, 1)   (op x : entrywise product)
# o = sigmoid(W_xo*x + W_ho*h + b_o)        (1, H, 1)
# h_new = o x tanh(C_new)                   (1, H, 1)   (op x : entrywise product)
# y = tanh(W_hy*h + b_y)                    (1, D, 1)
class LSTM(nn.Module):
    def __init__(self, hidden_size: int, emb_size: int):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size

        self.W_xi = nn.Parameter(xavier_normal_(torch.empty(1, hidden_size, emb_size)),     # (1, H, D)
                                 requires_grad=True).float()
        self.W_hi = nn.Parameter(xavier_normal_(torch.empty(1, hidden_size, hidden_size)),  # (1, H, H)
                                 requires_grad=True).float()
        self.B_i = nn.Parameter(xavier_normal_(torch.empty(1, hidden_size, 1)),             # (1, H, 1)
                                requires_grad=True).float()

        self.W_xg = nn.Parameter(xavier_normal_(torch.empty(1, hidden_size, emb_size)),     # (1, H, D)
                                 requires_grad=True).float()
        self.W_hg = nn.Parameter(xavier_normal_(torch.empty(1, hidden_size, hidden_size)),  # (1, H, H)
                                 requires_grad=True).float()
        self.B_g = nn.Parameter(xavier_normal_(torch.empty(1, hidden_size, 1)),             # (1, H, 1)
                                requires_grad=True).float()

        self.W_xf = nn.Parameter(xavier_normal_(torch.empty(1, hidden_size, emb_size)),     # (1, H, D)
                                 requires_grad=True).float()
        self.W_hf = nn.Parameter(xavier_normal_(torch.empty(1, hidden_size, hidden_size)),  # (1, H, H)
                                 requires_grad=True).float()
        self.B_f = nn.Parameter(xavier_normal_(torch.empty(1, hidden_size, 1)),             # (1, H, 1)
                                requires_grad=True).float()

        self.W_xo = nn.Parameter(xavier_normal_(torch.empty(1, hidden_size, emb_size)),     # (1, H, D)
                                 requires_grad=True).float()
        self.W_ho = nn.Parameter(xavier_normal_(torch.empty(1, hidden_size, hidden_size)),  # (1, H, H)
                                 requires_grad=True).float()
        self.B_o = nn.Parameter(xavier_normal_(torch.empty(1, hidden_size, 1)),             # (1, H, 1)
                                requires_grad=True).float()

        self.W_hy = nn.Parameter(xavier_normal_(torch.empty(1, emb_size, hidden_size)),     # (1, D, H)
                                 requires_grad=True).float()
        self.B_y = nn.Parameter(xavier_normal_(torch.empty(1, emb_size, 1)),                # (1, D, 1)
                                requires_grad=True).float()

    def get_initial_c_and_h(self, batch_size: int):
        c = torch.from_numpy(np.zeros([batch_size, self.hidden_size, 1])).float()
        h = torch.from_numpy(np.zeros([batch_size, self.hidden_size, 1])).float()
        return c, h

    def forward(self, x: Tensor, c: Tensor, h: Tensor):
        x_mul = torch.matmul(self.W_xi, x)
        h_mul = torch.matmul(self.W_hi, h)
        i = torch.sigmoid(x_mul + h_mul + self.B_i)

        x_mul = torch.matmul(self.W_xg, x)
        h_mul = torch.matmul(self.W_hg, h)
        g = torch.tanh(x_mul + h_mul + self.B_g)

        x_mul = torch.matmul(self.W_xf, x)
        h_mul = torch.matmul(self.W_hf, h)
        f = torch.sigmoid(x_mul + h_mul + self.B_f)

        c = torch.mul(f, c) + torch.mul(i, g)

        x_mul = torch.matmul(self.W_xo, x)
        h_mul = torch.matmul(self.W_ho, h)
        o = torch.sigmoid(x_mul + h_mul + self.B_o)

        h = torch.mul(o, c)

        y = torch.tanh(torch.matmul(self.W_hy, h) + self.B_y)

        return y, c, h







