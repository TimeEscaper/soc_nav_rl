# Based on https://github.com/vita-epfl/CrowdNav/blob/master/crowd_nav/utils/explorer.py

import logging
import copy
import torch

from typing import Optional
from stable_baselines3.common.vec_env import SubprocVecEnv


class Explorer:
    def __init__(self, env, device, memory=None, gamma=None):
        self._env = env
        self.device = device
        self.memory = memory
        self.gamma = gamma
        self.target_model = None

    def update_target_model(self, target_model):
        self.target_model = copy.deepcopy(target_model)

    def run_k_episodes(self, k, update_memory=False, imitation_learning=False, episode=None,
                       print_failure=False, policy: Optional = None, val: bool = False):
        success_times = []
        collision_times = []
        timeout_times = []
        success = 0
        collision = 0
        timeout = 0
        too_close = 0
        min_dist = []
        collision_cases = []
        timeout_cases = []

        states = {i: [] for i in range(self._env.num_envs)}
        rewards = {i: [] for i in range(self._env.num_envs)}
        episode_cnt = 0
        mean_cumulative_reward = 0.
        n_successes = 0

        ob = self._env.reset()
        if policy is not None:
            policy.reset(env_idx=None)

        while episode_cnt < k:
            if policy is None:
                ob, reward, done, info = self._env.step([None for _ in range(self._env.num_envs)])
            else:
                actions = policy.predict(ob, val=val)
                ob, reward, done, info = self._env.step(actions)

            for i in range(self._env.num_envs):
                states[i].append({k: v[i] for k, v in ob.items()})
                rewards[i].append(reward[i])

                if done[i]:
                    done_reason = info[i]["done_reason"]
                    if update_memory and done_reason in ("success", "collision"):
                        self.update_memory(states[i], rewards[i], imitation_learning)
                    if done_reason == "success":
                        n_successes += 1
                    mean_cumulative_reward += sum(rewards[i])
                    states[i] = []
                    rewards[i] = []
                    episode_cnt += 1
                    ob_local = self._env.env_method("reset", indices=i)[0]
                    for key in ob.keys():
                        ob[key][i] = ob_local[key]
                    if policy is not None:
                        policy.reset(env_idx=i)
                    print(f"Finished episode {episode_cnt} out of {k}")

        print(f"Total replay memory size: {len(self.memory)}")

        return mean_cumulative_reward / episode_cnt, n_successes / episode_cnt

        #     ob = self._env.reset()
        #     done = False
        #     states = []
        #     rewards = []
        #     info = {}
        #     while not done:
        #         if isinstance(self._env, SubprocVecEnv):
        #             action = [None for _ in range(self._env.num_envs)]
        #         else:
        #             action = None
        #         ob, reward, done, info = self._env.step(action)
        #         states.append(ob)
        #         rewards.append(reward)
        #         #
        #         # if isinstance(info, Danger):
        #         #     too_close += 1
        #         #     min_dist.append(info.min_dist)
        #
        #     done_reason = info["done_reason"] if "done_reason" in info else None
        #     if done_reason is None:
        #         raise ValueError("Invalid end signal from environment")
        #
        #     if done_reason == "success":
        #         success += 1
        #         # success_times.append(self._env.global_time)
        #     elif done_reason == "collision":
        #         collision += 1
        #         collision_cases.append(i)
        #     elif done_reason == "truncated":
        #         timeout += 1
        #         timeout_cases.append(i)
        #         timeout_times.append(self._env.time_limit)
        #     else:
        #         raise ValueError('Invalid end signal from environment')
        #
        #     if update_memory:
        #         if done_reason == "success" or done_reason == "collision":
        #             # only add positive(success) or negative(collision) experience in experience set
        #             self.update_memory(states, rewards, imitation_learning)
        #
        #     cumulative_rewards.append(sum([pow(self.gamma, t)
        #                                    * reward for t, reward in enumerate(rewards)]))
        #     # cumulative_rewards.append(sum([pow(self.gamma, t * self.robot.time_step * self.robot.v_pref)
        #     #                                * reward for t, reward in enumerate(rewards)]))
        #
        # success_rate = success / k
        # collision_rate = collision / k
        # assert success + collision + timeout == k
        # # avg_nav_time = sum(success_times) / len(success_times) if success_times else self._env.time_limit
        #
        # extra_info = '' if episode is None else 'in episode {} '.format(episode)
        # logging.info('{}has success rate: {:.2f}, collision rate: {:.2f}, total reward: {:.4f}'.
        #              format(extra_info, success_rate, collision_rate,
        #                     average(cumulative_rewards)))
        # # if phase in ['val', 'test']:
        # #     num_step = sum(success_times + collision_times + timeout_times) / self.robot.time_step
        # #     logging.info('Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
        # #                  too_close / num_step, average(min_dist))
        #
        # if print_failure:
        #     logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
        #     logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))

    def update_memory(self, states, rewards, imitation_learning=False):
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')

        for i, state in enumerate(states):
            reward = rewards[i]

            # VALUE UPDATE
            if imitation_learning:
                # define the value of states in IL as cumulative discounted rewards, which is the same in RL
                # state = self.target_policy.transform(state)
                # value = pow(self.gamma, (len(states) - 1 - i) * self.robot.time_step * self.robot.v_pref)
                value = sum([pow(self.gamma, max(t - i, 0)) * reward
                             * (1 if t >= i else 0) for t, reward in enumerate(rewards)])
            else:
                if i == len(states) - 1:
                    # terminal state
                    value = reward
                else:
                    next_state = states[i + 1]
                    gamma_bar = pow(self.gamma, 1.)
                    next_state = {k: torch.Tensor(v).unsqueeze(0).to(self.device) for k, v in next_state.items()}
                    with torch.no_grad():
                        value = self.target_model(next_state).item()
                    value = reward + gamma_bar * value
            value = torch.Tensor([value]).float()

            self.memory.push((state, value))


def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0
