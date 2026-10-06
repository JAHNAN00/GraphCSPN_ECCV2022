"""Preserve the released NYU graph and fixed geometry values."""
from types import SimpleNamespace

from torch import nn

MODEL_NAME = 'GraphCSPN'
DCN_BACKEND = 'none (dynamic KNN graph)'


def model_config(seed, data_dir):
    return SimpleNamespace(seed=seed, from_scratch=True, prop_time=1, network='resnet34',
                           knn_neighbors=16, graph_hidden_channels=96, graph_layers=3,
                           input_hw=[228, 304], graph_hw=[76, 102],
                           geometry='original fixed NYU camera values; GPU buffers, no H2D in forward')


def build_model(args):
    from model.graphcspn import GraphCSPN
    return GraphCSPN(args)


class DepthOnly(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, rgb, dep):
        return self.net({'rgb': rgb, 'dep': dep})


def reference_prediction(model, rgb, dep):
    return model.net({'rgb': rgb, 'dep': dep})
