import torch
import torch.nn as nn
import gym


class BasicGraphExtractor(nn.Module):

    def __init__(self, observation_space: gym.spaces.Dict):
        super(BasicGraphExtractor, self).__init__()
