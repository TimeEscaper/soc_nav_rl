# Based on https://github.com/vita-epfl/CrowdNav/blob/master/crowd_nav/utils/explorer.py

import logging
import copy
import torch


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
                       print_failure=False):
        success_times = []
        collision_times = []
        timeout_times = []
        success = 0
        collision = 0
        timeout = 0
        too_close = 0
        min_dist = []
        cumulative_rewards = []
        collision_cases = []
        timeout_cases = []
        for i in range(k):
            ob = self._env.reset()
            done = False
            states = []
            rewards = []
            info = {}
            while not done:
                ob, reward, done, info = self._env.step(None)
                states.append(ob)
                rewards.append(reward)
                #
                # if isinstance(info, Danger):
                #     too_close += 1
                #     min_dist.append(info.min_dist)

            done_reason = info["done_reason"] if "done_reason" in info else None
            if done_reason is None:
                raise ValueError("Invalid end signal from environment")

            if done_reason == "success":
                success += 1
                # success_times.append(self._env.global_time)
            elif done_reason == "collision":
                collision += 1
                collision_cases.append(i)
            elif done_reason == "truncated":
                timeout += 1
                timeout_cases.append(i)
                timeout_times.append(self._env.time_limit)
            else:
                raise ValueError('Invalid end signal from environment')

            if update_memory:
                if done_reason == "success" or done_reason == "collision":
                    # only add positive(success) or negative(collision) experience in experience set
                    self.update_memory(states, rewards, imitation_learning)

            cumulative_rewards.append(sum([pow(self.gamma, t)
                                           * reward for t, reward in enumerate(rewards)]))
            # cumulative_rewards.append(sum([pow(self.gamma, t * self.robot.time_step * self.robot.v_pref)
            #                                * reward for t, reward in enumerate(rewards)]))

        success_rate = success / k
        collision_rate = collision / k
        assert success + collision + timeout == k
        # avg_nav_time = sum(success_times) / len(success_times) if success_times else self._env.time_limit

        extra_info = '' if episode is None else 'in episode {} '.format(episode)
        logging.info('{}has success rate: {:.2f}, collision rate: {:.2f}, total reward: {:.4f}'.
                     format(extra_info, success_rate, collision_rate,
                            average(cumulative_rewards)))
        # if phase in ['val', 'test']:
        #     num_step = sum(success_times + collision_times + timeout_times) / self.robot.time_step
        #     logging.info('Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
        #                  too_close / num_step, average(min_dist))

        if print_failure:
            logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
            logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))

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
                    value = reward + gamma_bar * self.target_model(next_state.unsqueeze(0)).data.item()
            value = torch.Tensor([value]).float()

            self.memory.push((state, value))


def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0
