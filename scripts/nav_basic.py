import fire
import nip
import gym
import numpy as np

from typing import Optional
from functools import partial
from pathlib import Path
from nip.elements import Element
from lib.envs import AbstractEnvFactory


def _train(output_dir: str,
           train_env_factory: AbstractEnvFactory,
           config: Element,
           **_):
    train_env = train_env_factory()

    obs = train_env.reset()
    done = False
    while not done:
        obs, reward, done, info = train_env.step(train_env.action_space.sample())


def main(config: str, output_dir: Optional[str] = "."):
    nip.run(config, partial(_train, output_dir=output_dir),
            verbose=False, return_configs=False, config_parameter='config', nonsequential=True)


if __name__ == '__main__':
    fire.Fire(main)
