import fire
import gym
import nip
import torch

from datetime import datetime
from typing import Optional, Callable, Dict, Any
from functools import partial
from pathlib import Path
from nip.elements import Element
# from stable_baselines3.common.vec_env import SubprocVecEnv
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


def _make_subproc_env(env_factory: Callable, n_proc: int) -> SubprocVecEnvCustom:
    return SubprocVecEnvCustom([env_factory for _ in range(n_proc)])


def _train(output_dir: str,
           experiment_name: str,
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
    prefix = f"{experiment_name}__" if experiment_name is not None else ""
    output_dir = Path(output_dir) / f"{prefix}{datetime.today().strftime('%Y_%m_%d__%H_%M_%S')}"

    # train_env = train_env_factory(is_eval=False)
    train_env = _make_subproc_env(lambda: train_env_factory(is_eval=False), n_proc=n_train_envs)
    # eval_env = Monitor(EvalEnvWrapper(eval_env_factory() if eval_env_factory is not None else
    #                                   train_env_factory(is_eval=True),
    #                                   curriculum,
    #                                   n_eval_episodes=eval_n_episodes,
    #                                   logger=logger,
    #                                   train_env=train_env))

    output_dir.mkdir(parents=True)
    config_path = output_dir / "config.yaml"
    nip.dump(config_path, config)

    rl_model = DeepVLearning(
        train_env=train_env,
        logger=logger,
        **v_learning_params
    )

    logger.init()
    logger.log("experiment_id", str(output_dir.name))
    logger.log("seed", str(seed))
    logger.upload_config(config_path)

    rl_model.imitation_learning(il_config, output_dir)

    train_env.close()
    logger.close()


def main(config: str,
         output_dir: Optional[str] = "./experiments",
         experiment_name: Optional[str] = None,
         seed: int = 42):
    config = Path(config)
    nip.run(config, partial(_train, output_dir=output_dir, experiment_name=experiment_name, seed=seed),
            verbose=False, return_configs=False, config_parameter='config', nonsequential=True)


if __name__ == '__main__':
    fire.Fire(main)
