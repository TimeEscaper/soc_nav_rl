import gym
import numpy as np

from lib.envs.curriculum import AbstractCurriculum
from lib.utils import AbstractLogger


class EvalEnvWrapper(gym.Env):

    def __init__(self, env: gym.Env, curriculum: AbstractCurriculum, n_eval_episodes: int, logger: AbstractLogger):
        self._env = env
        self._curriculum = curriculum
        self._n_eval_episodes = n_eval_episodes
        self._logger = logger
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self._current_episode = 0
        self._current_cumulative_reward = 0.
        self._current_cumulative_reward_components = {}

        self._cumulative_rewards_histories = []
        self._cumulative_rewards_components_histories = {}
        self._success_histories = []

    def step(self, action: np.ndarray):
        obs, reward, done, info = self._env.step(action)

        self._current_cumulative_reward += reward
        if "reward" in info:
            for k, v in info["reward"].items():
                if k in self._current_cumulative_reward_components:
                    self._current_cumulative_reward_components[k] += v
                else:
                    self._current_cumulative_reward_components[k] = v

        if done:
            success = info["done_reason"] == "success"
            self._success_histories.append(success)
            self._cumulative_rewards_histories.append(self._current_cumulative_reward)

            for component, cumulative_reward in self._current_cumulative_reward_components.items():
                if component in self._cumulative_rewards_components_histories:
                    self._cumulative_rewards_components_histories[component].append(cumulative_reward)
                else:
                    self._cumulative_rewards_components_histories[component] = [cumulative_reward]
            self._current_episode += 1

            if self._current_episode == self._n_eval_episodes:
                mean_cumulative_reward = np.mean(self._cumulative_rewards_histories)
                success_ratio = np.array(self._success_histories).sum() / len(self._success_histories)
                self._logger.log("eval/mean_cumulative_reward", mean_cumulative_reward)
                self._logger.log("eval/success_ratio", success_ratio)
                for component, rewards in self._cumulative_rewards_components_histories.items():
                    self._logger.log(f"eval/mean_reward_components/{component}", np.mean(rewards))

                threshold = self._curriculum.get_success_rate_threshold()
                if threshold is not None:
                    if success_ratio >= threshold:
                        self._curriculum.update_stage()
                current_stage, _ = self._curriculum.get_current_stage()
                self._logger.log("stage_idx", current_stage)

                self._current_episode = 0
                self._success_histories = []
                self._cumulative_rewards_histories = []
                self._cumulative_rewards_components_histories = {}

            self._current_cumulative_reward = 0
            self._current_cumulative_reward_components = {}

        return obs, reward, done, info

    def reset(self):
        return self._env.reset()

    def render(self, mode="human"):
        self._env.render(mode)
