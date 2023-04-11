# Based on https://github.com/vita-epfl/CrowdNav/blob/master/crowd_nav/train.py

import gym
import numpy as np
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
from lib.rl.v_learning.policy import MaxVSubgoalPolicy
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


@dataclass
@nip
class RLConfig:
    train_epochs: int
    batch_size: int
    lr: float
    wd: float = 0.
    target_update_interval: int = 50
    replay_capacity: int = 100000
    sample_episodes: int = 1
    train_batches: int = 100
    epsilon_start: float = 0.5
    epsilon_end: float = 0.1
    epsilon_decay: int = 4000
    eval_period: int = 100
    eval_episodes: int = 10
    device: str = "cuda"


@nip
class DeepVLearning:

    def __init__(self,
                 train_env: Union[gym.Env, VecEnv],
                 logger: AbstractLogger,
                 policy_kwargs: Dict[str, Any],
                 vf_kwargs: Optional[Dict[str, Any]] = None,
                 eval_env: Optional[gym.Env] = None,
                 gamma: float = 0.99,
                 ):
        self._policy_kwargs = policy_kwargs
        self._model = ValueNetwork(
            feature_extractor=vf_kwargs["feature_extractor"](train_env.observation_space,
                                                             **vf_kwargs["feature_extractor_kwargs"]),
            activation_fn=vf_kwargs["activation_fn"],
            net_arch=vf_kwargs["net_arch"]
        )
        self._train_env = train_env
        self._eval_env = eval_env
        self._logger = logger
        self._gamma = gamma

    @property
    def value_network(self) -> nn.Module:
        return self._model

    def imitation_learning(self, il_config: ILConfig, output_path: Path):
        device = self._get_device(il_config.device)

        self._model = self._model.to(device)
        _ = self._model.train()

        memory = ReplayMemory(il_config.replay_capacity)
        trainer = SupervisedTrainer(self._model, memory, device, il_config.batch_size)
        trainer.set_learning_rate(il_config.lr, il_config.wd)
        explorer = Explorer(self._train_env, device, memory, self._gamma)

        explorer.run_k_episodes(il_config.collect_episodes, update_memory=True, imitation_learning=True)
        trainer.optimize_epoch(il_config.train_epochs,
                               str(output_path / "model_il.pth.tar"), self._logger)

    def reinforcement_learning(self, rl_config: RLConfig, output_path: Path):
        device = self._get_device(rl_config.device)

        self._model = self._model.to(device)
        _ = self._model.train()

        memory = ReplayMemory(rl_config.replay_capacity)
        trainer = SupervisedTrainer(self._model, memory, device, rl_config.batch_size)
        trainer.set_learning_rate(rl_config.lr, rl_config.wd)
        explorer = Explorer(self._train_env, device, memory, self._gamma)
        explorer_eval = Explorer(self._eval_env, device, memory, self._gamma)

        policy = MaxVSubgoalPolicy(n_envs=self._train_env.num_envs,
                                   device=device,
                                   **self._policy_kwargs)
        policy.value_network = self._model

        eval_kwargs = self._policy_kwargs.copy()
        policy_eval = MaxVSubgoalPolicy(n_envs=self._eval_env.num_envs,
                                        device=device,
                                        **eval_kwargs)
        policy_eval.value_network = self._model

        epoch_cnt = 0
        best_val_reward = -np.inf

        explorer.update_target_model(self._model)
        explorer_eval.update_target_model(self._model)

        while epoch_cnt < rl_config.train_epochs:
            if epoch_cnt < rl_config.epsilon_decay:
                epsilon = rl_config.epsilon_start + (
                        rl_config.epsilon_end - rl_config.epsilon_start) / rl_config.epsilon_decay * epoch_cnt
            else:
                epsilon = rl_config.epsilon_end
            policy.eps = epsilon

            # evaluate the model
            if epoch_cnt % rl_config.eval_period == 0:
                val_reward, success_ratio = explorer_eval.run_k_episodes(rl_config.eval_episodes, update_memory=False,
                                                          imitation_learning=False,
                                                          val=True, policy=policy_eval)
                self._logger.log("rl/val_reward", val_reward)
                self._logger.log("rl/val_success_ratio", success_ratio)
                if val_reward > best_val_reward:
                    best_val_reward = val_reward
                    torch.save(self._model.state_dict(), str(output_path / "model_rl.pth.tar"))

            # sample k episodes into memory and optimize over the generated memory
            explorer.run_k_episodes(rl_config.sample_episodes, update_memory=True, imitation_learning=False, val=False,
                                    policy=policy)
            if len(memory) > 0:
                trainer.optimize_batch(rl_config.train_batches)
                epoch_cnt += 1

                if epoch_cnt % rl_config.target_update_interval == 0:
                    explorer.update_target_model(self._model)
                    explorer_eval.update_target_model(self._model)

    def load_il(self, model_path: Path):
        self._model.load_state_dict(torch.load(str(model_path)))

    def _get_device(self, device: str) -> str:
        device = device
        if device != "cpu":
            if not torch.cuda.is_available():
                device = "cpu"
        return device
