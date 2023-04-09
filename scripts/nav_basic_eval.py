import fire
import gym
import nip

from datetime import datetime
from typing import Optional, Callable, Dict, Any
from functools import partial
from pathlib import Path
from nip.elements import Element
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib import RecurrentPPO

from lib.envs import AbstractEnvFactory
from lib.envs.curriculum import AbstractCurriculum
from lib.envs.wrappers import EvalEnvWrapper
from lib.utils import AbstractLogger, ConsoleLogger
from lib.rl.callbacks import CustomEvalCallback
from lib.utils.sampling import seed_all


def _make_subproc_env(env_factory: Callable, n_proc: int) -> SubprocVecEnv:
    return SubprocVecEnv([env_factory for _ in range(n_proc)])


def _eval(config: Element,
          stage: int,
          seed: int,
          rl_model: Any,
          model_path: Path,
          curriculum: AbstractCurriculum,
          train_env_factory: Optional[AbstractEnvFactory] = None,
          eval_env_factory: Optional[Callable] = None,
          **_):
    seed_all(seed)

    for _ in range(stage):
        curriculum.update_stage()

    rl_model = rl_model.load(str(model_path))
    if eval_env_factory is not None:
        eval_env = eval_env_factory()
    elif train_env_factory is not None:
        eval_env = train_env_factory(is_eval=True)
    else:
        raise ValueError("Train env or eval env must be set")
    eval_env.enable_render()

    evaluate_policy(
        rl_model,
        eval_env,
        n_eval_episodes=10,
        render=False,
        deterministic=True,
        return_episode_rewards=True,
        warn=True
    )

    # while True:
    #     done = False
    #     states = None
    #     obs = eval_env.reset()
    #     while not done:
    #         action, states = rl_model.predict(obs, state=states, deterministic=True)
    #         obs, reward, done, info = eval_env.step(action)


def main(result_dir: str,
         seed: int = 42,
         stage: Optional[int] = None):
    result_dir = Path(result_dir)

    model_files = sorted(result_dir.glob("*.zip"))
    if stage is None:
        stage = len(model_files) - 1

    model_file = model_files[stage]
    print(f"Evaluating {model_file.name} (stage {stage})")

    nip.run(result_dir / "config.yaml", partial(_eval, model_path=model_file, seed=seed, stage=stage),
            verbose=False, return_configs=False, config_parameter='config', nonsequential=True)


if __name__ == '__main__':
    fire.Fire(main)
