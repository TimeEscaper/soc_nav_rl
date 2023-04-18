from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Optional, Callable, Dict, Any

import fire
import gym
import nip
from nip.elements import Element
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from lib.envs.curriculum import AbstractCurriculum
from lib.envs.task_wrappers import WrappedEnvFactory
from lib.envs.util_wrappers import EvalEnvWrapper
from lib.rl.callbacks import CustomEvalCallback
from lib.utils import AbstractLogger, ConsoleLogger
from lib.utils.layers import get_activation
from lib.utils.sampling import seed_all


def _make_subproc_env(env_factory: Callable, n_proc: int) -> SubprocVecEnv:
    return SubprocVecEnv([env_factory for _ in range(n_proc)])


def _train(output_dir: str,
           experiment_name: str,
           seed: int,
           train_env_factory: WrappedEnvFactory,
           n_train_envs: int,
           eval_period: int,
           eval_n_episodes: int,
           rl_model: Any,
           curriculum: AbstractCurriculum,
           config: Element,
           rl_model_params: Optional[Dict[str, Any]] = None,
           eval_env_factory: Optional[Callable] = None,
           logger: AbstractLogger = ConsoleLogger(),
           feature_extractor: Optional = None,
           feature_extractor_kwargs: Optional[Dict[str, Any]] = None,
           **_):
    seed_all(seed)
    prefix = f"{experiment_name}__" if experiment_name is not None else ""
    output_dir = Path(output_dir) / f"{prefix}{datetime.today().strftime('%Y_%m_%d__%H_%M_%S')}"

    train_env = _make_subproc_env(lambda: train_env_factory(is_eval=False), n_proc=n_train_envs)
    train_env.reset()

    eval_env = Monitor(EvalEnvWrapper(eval_env_factory() if eval_env_factory is not None else
                                      train_env_factory(is_eval=True),
                                      curriculum,
                                      n_eval_episodes=eval_n_episodes,
                                      logger=logger,
                                      train_env=train_env))

    output_dir.mkdir(parents=True)
    tensorboard_dir = output_dir / "tensorboard"
    config_path = output_dir / "config.yaml"
    nip.dump(config_path, config)

    eval_callback = CustomEvalCallback(eval_env=eval_env,
                                       curriculum=curriculum,
                                       n_eval_episodes=eval_n_episodes,
                                       eval_freq=eval_period,
                                       best_model_save_path=str(output_dir),
                                       deterministic=True,
                                       verbose=1)

    rl_model_params = rl_model_params or {}
    if feature_extractor is not None:
        if "policy_kwargs" in rl_model_params:
            rl_model_params["policy_kwargs"]["features_extractor_class"] = feature_extractor
        else:
            rl_model_params["policy_kwargs"] = {"features_extractor_class": feature_extractor}
        if feature_extractor_kwargs is not None:
            rl_model_params["policy_kwargs"]["features_extractor_kwargs"] = feature_extractor_kwargs
    if "policy_kwargs" in rl_model_params and "activation_fn" in rl_model_params["policy_kwargs"]:
        rl_model_params["policy_kwargs"]["activation_fn"] = get_activation(
            rl_model_params["policy_kwargs"]["activation_fn"])

    if rl_model != RecurrentPPO:
        policy = "MultiInputPolicy" if isinstance(train_env.observation_space, gym.spaces.Dict) else "MlpPolicy"
    else:
        policy = "MultiInputLstmPolicy" if isinstance(train_env.observation_space, gym.spaces.Dict) else "MlpLstmPolicy"
    rl_model = rl_model(
        policy=policy,
        env=train_env,
        tensorboard_log=str(tensorboard_dir),
        **rl_model_params
    )

    logger.init()
    logger.log("experiment_id", str(output_dir.name))
    logger.log("seed", str(seed))
    logger.upload_config(config_path)
    rl_model.learn(int(1e9), callback=eval_callback)

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
