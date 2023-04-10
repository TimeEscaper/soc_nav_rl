import fire
import gym
import nip
import numpy as np
import torch

from datetime import datetime
from typing import Optional, Callable, Dict, Any
from functools import partial
from pathlib import Path
from nip.elements import Element
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import Monitor
from sb3_contrib import RecurrentPPO

from lib.envs import AbstractEnvFactory
from lib.envs.curriculum import AbstractCurriculum
from lib.envs.wrappers import EvalEnvWrapper
from lib.utils import AbstractLogger, ConsoleLogger
from lib.rl.callbacks import CustomEvalCallback
from lib.utils.sampling import seed_all
from lib.utils.layers import get_activation
from lib.rl.v_learning.deep_v_learning import DeepVLearning, ILConfig
from lib.rl.v_learning.policy import MaxVSubgoalPolicy


def _eval(result_dir: Path,
          seed: int,
          train_env_factory: AbstractEnvFactory,
          n_train_envs: int,
          eval_period: int,
          eval_n_episodes: int,
          curriculum: AbstractCurriculum,
          il_config: ILConfig,
          config: Element,
          v_learning_params: Optional[Dict[str, Any]] = None,
          eval_env_factory: Optional[Callable] = None,
          logger: AbstractLogger = ConsoleLogger(),
          **_):
    seed_all(seed)

    eval_env = train_env_factory(is_eval=True) if eval_env_factory is None else eval_env_factory()
    eval_env.enable_render()

    rl_model = DeepVLearning(
        train_env=eval_env,
        logger=logger,
        **v_learning_params
    )
    value_network = rl_model.value_network
    value_network.load_state_dict(torch.load(str(result_dir / "model_il.pth.tar"), map_location="cuda"))
    policy = MaxVSubgoalPolicy(subgoal_linear=np.array([0., 1., 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.]),
                               subgoal_angular=np.deg2rad(np.linspace(-110., 110, 9)),
                               max_linear_vel=2.,
                               dt=0.1,
                               device="cuda")
    value_network = value_network.to(device="cuda")
    _ = value_network.eval()
    policy.value_network = value_network

    while True:
        obs = eval_env.reset()
        policy.reset()
        done = False
        while not done:
            action = policy.predict(obs)
            obs, reward, done, info = eval_env.step(action)

    eval_env.close()


def main(result_dir: str,
         seed: int = 42):
    result_dir = Path(result_dir)
    config = result_dir / "config.yaml"
    nip.run(config, partial(_eval, result_dir=result_dir, seed=seed),
            verbose=False, return_configs=False, config_parameter='config', nonsequential=True)


if __name__ == '__main__':
    fire.Fire(main)
