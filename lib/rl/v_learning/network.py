import torch
import torch.nn as nn

from typing import Union, Optional, Callable, List
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from lib.utils.layers import get_activation


class ValueNetwork(nn.Module):

    def __init__(self,
                 feature_extractor: BaseFeaturesExtractor,
                 activation_fn: Union[str, Callable] = nn.Tanh,
                 net_arch: Optional[List[int]] = None):
        if net_arch is not None:
            assert len(net_arch) > 0, f"net_arch must not be empty if specified"
        super(ValueNetwork, self).__init__()

        if isinstance(activation_fn, str):
            activation_fn = get_activation(activation_fn)
        if net_arch is None:
            net_arch = [feature_extractor.features_dim]

        self._feature_extractor = feature_extractor

        head = []
        previous_dim = feature_extractor.features_dim
        for dim in net_arch:
            head.append(nn.Linear(previous_dim, dim))
            head.append(activation_fn())
            previous_dim = dim
        head.append(nn.Linear(previous_dim, 1))
        self._head = nn.Sequential(*head)

    def forward(self, observation) -> torch.Tensor:
        feature = self._feature_extractor(observation)
        value = self._head(feature)
        return value
