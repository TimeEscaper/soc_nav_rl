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
from lib.envs.subproc_custom import SubprocVecEnvCustom
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


def _make_subproc_env(env_factory: Callable, n_proc: int) -> SubprocVecEnvCustom:
    return SubprocVecEnvCustom([env_factory for _ in range(n_proc)])


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

    eval_env = _make_subproc_env(
        lambda: train_env_factory(is_eval=True) if eval_env_factory is None else eval_env_factory(), n_proc=1)
    eval_env.env_method("enable_render")

    if v_learning_params is not None:
        v_learning_params["policy_kwargs"] = {
            "subgoal_linear": np.array([0., 1., 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.]),
            "subgoal_angular": np.deg2rad(np.linspace(-110., 110, 9)),
            "max_linear_vel":  2.,
            "dt": 0.1,
            "device": "cpu",
            "n_envs": eval_env.num_envs
        }
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
                               n_envs=eval_env.num_envs,
                               device="cpu")
    value_network = value_network.to(device="cpu")
    _ = value_network.eval()
    policy.value_network = value_network

    obs = eval_env.reset()
    policy.reset(env_idx=None)
    envs_to_reset = []
    while True:
        for env_idx in envs_to_reset:
            env_obs = eval_env.env_method("reset", indices=env_idx)[0]
            policy.reset(env_idx=env_idx)
            for k in obs.keys():
                obs[k][env_idx] = env_obs[k]
        envs_to_reset = []

        action = policy.predict(obs)
        obs, reward, done, info = eval_env.step(action)
        for i in range(done.shape[0]):
            if done[i]:
                envs_to_reset.append(i)

    eval_env.close()


def main(result_dir: str,
         seed: int = 42):
    result_dir = Path(result_dir)
    config = result_dir / "config.yaml"
    nip.run(config, partial(_eval, result_dir=result_dir, seed=seed),
            verbose=False, return_configs=False, config_parameter='config', nonsequential=True)


if __name__ == '__main__':
    fire.Fire(main)
