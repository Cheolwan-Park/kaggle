import torch


def xavier(param):
    torch.nn.init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        xavier(m.weight.data)