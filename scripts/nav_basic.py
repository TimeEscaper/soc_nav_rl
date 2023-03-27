import fire
import nip

from datetime import datetime
from typing import Optional, Callable, Dict, Any
from functools import partial
from pathlib import Path
from nip.elements import Element
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import PPO

from lib.envs import AbstractEnvFactory
from lib.envs.wrappers import EvalEnvWrapper
from lib.utils import AbstractLogger, ConsoleLogger


def _make_subproc_env(env_factory: Callable, n_proc: int) -> SubprocVecEnv:
    return SubprocVecEnv([env_factory for _ in range(n_proc)])


def _train(output_dir: str,
           experiment_name: str,
           train_env_factory: AbstractEnvFactory,
           n_train_envs: int,
           eval_period: int,
           eval_n_episodes: int,
           config: Element,
           rl_model_params: Optional[Dict[str, Any]] = None,
           eval_env_factory: Optional[Callable] = None,
           logger: AbstractLogger = ConsoleLogger(),
           **_):
    prefix = f"{experiment_name}__" if experiment_name is not None else ""
    output_dir = Path(output_dir) / f"{prefix}{datetime.today().strftime('%Y_%m_%d__%H_%M_%S')}"

    train_env = _make_subproc_env(train_env_factory, n_proc=n_train_envs)
    eval_env = Monitor(EvalEnvWrapper(eval_env_factory() if eval_env_factory is not None else train_env_factory(),
                                      n_eval_episodes=eval_n_episodes,
                                      logger=logger))

    output_dir.mkdir(parents=True)
    nip.dump(output_dir / "config.nip", config)

    eval_callback = EvalCallback(eval_env=eval_env,
                                 n_eval_episodes=eval_n_episodes,
                                 eval_freq=eval_period,
                                 best_model_save_path=str(output_dir),
                                 deterministic=True,
                                 verbose=1)
    rl_model_params = rl_model_params or {}
    rl_model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        **rl_model_params
    )

    logger.init()
    rl_model.learn(int(1e9), callback=eval_callback)

    train_env.close()
    logger.close()


def _eval(config: Element,
          model_path: Path,
          train_env_factory: Optional[AbstractEnvFactory] = None,
          eval_env_factory: Optional[Callable] = None,
          **_):
    model = PPO.load(str(model_path))
    if eval_env_factory is not None:
        eval_env = eval_env_factory()
    elif train_env_factory is not None:
        eval_env = train_env_factory()
    else:
        raise ValueError("Train env or eval env must be set")
    eval_env.enable_render()

    while True:
        done = False
        obs = eval_env.reset()
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = eval_env.step(action)


def main(config: str, output_dir: Optional[str] = "./experiments", experiment_name: Optional[str] = None):
    config = Path(config)
    if config.is_file():
        nip.run(config, partial(_train, output_dir=output_dir, experiment_name=experiment_name),
                verbose=False, return_configs=False, config_parameter='config', nonsequential=True)
    elif config.is_dir():
        nip.run(config / "config.nip", partial(_eval, model_path=config / "best_model.zip"),
                verbose=False, return_configs=False, config_parameter='config', nonsequential=True)


if __name__ == '__main__':
    fire.Fire(main)
