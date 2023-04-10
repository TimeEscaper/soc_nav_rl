# Based on https://github.com/vita-epfl/CrowdNav/blob/master/crowd_nav/train.py

import gym
import torch
import torch.nn as nn

from typing import Optional, Callable, Dict, List, Any, Union
from dataclasses import dataclass
from pathlib import Path
from nip import nip
from stable_baselines3.common.vec_env import VecEnv

from lib.envs.sim_config_samplers import AbstractActionSpaceConfig
from lib.rl.v_learning.network import ValueNetwork
from lib.rl.v_learning.replay_memory import ReplayMemory
from lib.rl.v_learning.explorer import Explorer
from lib.rl.v_learning.supervised_trainer import SupervisedTrainer
from lib.utils import AbstractLogger


@dataclass
@nip
class ILConfig:
    collect_episodes: int
    train_epochs: int
    batch_size: int
    lr: float
    wd: float = 0.
    replay_capacity: int = 100000
    device: str = "cuda"


@nip
class DeepVLearning:

    def __init__(self,
                 train_env: Union[gym.Env, VecEnv],
                 logger: AbstractLogger,
                 vf_kwargs: Optional[Dict[str, Any]] = None,
                 eval_env: Optional[gym.Env] = None,
                 n_eval_episodes: Optional[int] = 10,
                 gamma: float = 0.99):
        self._model = ValueNetwork(
            feature_extractor=vf_kwargs["feature_extractor"](train_env.observation_space,
                                                             **vf_kwargs["feature_extractor_kwargs"]),
            activation_fn=vf_kwargs["activation_fn"],
            net_arch=vf_kwargs["net_arch"]
        )
        self._train_env = train_env
        self._logger = logger
        self._gamma = gamma

    @property
    def value_network(self) -> nn.Module:
        return self._model

    def imitation_learning(self, il_config: ILConfig, output_path: Path):
        device = il_config.device
        if device != "cpu":
            if not torch.cuda.is_available():
                device = "cpu"

        self._model = self._model.to(device)
        _ = self._model.train()

        memory = ReplayMemory(il_config.replay_capacity)
        trainer = SupervisedTrainer(self._model, memory, device, il_config.batch_size)
        trainer.set_learning_rate(il_config.lr, il_config.wd)
        explorer = Explorer(self._train_env, device, memory, self._gamma)

        explorer.run_k_episodes(il_config.collect_episodes, update_memory=True, imitation_learning=True)
        trainer.optimize_epoch(il_config.train_epochs,
                               str(output_path / "model_il.pth.tar"), self._logger)

    def load_il(self):
        pass
