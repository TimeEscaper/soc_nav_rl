import numpy as np
import torch
import torch.nn as nn

from typing import Optional, Dict

from pyminisim.core import ROBOT_RADIUS, PEDESTRIAN_RADIUS
from nip import nip


@nip
class MaxVSubgoalPolicy:

    def __init__(self,
                 subgoal_linear: np.ndarray,
                 subgoal_angular: np.ndarray,
                 max_linear_vel: float,
                 dt: float,
                 n_envs: int,
                 device: str = "cuda",
                 eps: float = 0.):
        self._value_network: nn.Module = None
        self._subgoal_linear = subgoal_linear.copy()
        self._subgoal_angular = subgoal_angular.copy()
        self._max_linear_vel = max_linear_vel
        self._dt = dt
        self._n_envs = n_envs
        self._device = device
        self._step_cnt = np.array([0 for _ in range(n_envs)])
        self._eps = eps

    @property
    def value_network(self) -> nn.Module:
        return self._value_network

    @value_network.setter
    def value_network(self, network: nn.Module):
        self._value_network = network

    @property
    def eps(self) -> float:
        return self._eps

    @eps.setter
    def eps(self, value: float):
        self._eps = value

    def predict(self, observation: Dict[str, np.ndarray], val: bool = False) -> np.ndarray:
        if not val:
            random_action = np.random.choice([True, False], p=[self._eps, 1. - self._eps])
            if random_action:
                random_linear = np.random.choice(self._subgoal_linear, size=self._n_envs)
                random_angular = np.random.choice(self._subgoal_angular, size=self._n_envs)
                return np.stack([random_linear, random_angular], axis=1)

        possible_actions = []
        possible_values = []

        for subgoal_linear in self._subgoal_linear:
            for subgoal_angular in self._subgoal_angular:
                actions = np.array([subgoal_linear, subgoal_angular])
                values = self._eval_action(np.array([subgoal_linear, subgoal_angular]), observation)
                possible_actions.append(actions)
                possible_values.append(values)
                # print(action, value)
        possible_actions = np.array(possible_actions)
        possible_values = np.array(possible_values)

        max_values_indices = np.argmax(possible_values, axis=0)
        actions = possible_actions[max_values_indices]

        for i in range(self._n_envs):
            self._step_cnt[i] += 1

        # print(f"Action: {action}, value: {max_value}")
        return actions

    def _eval_action(self, subgoal_polar: np.ndarray, observation: Dict[str, np.ndarray]) -> float:
        robot_global = observation["robot_global"]
        subgoal_global = self._subgoal_to_global(subgoal_polar, observation["robot_global"])
        # print(subgoal_global)
        distance = np.linalg.norm(robot_global[:, :2] - subgoal_global, axis=1)

        approx_timesteps_indices = ((distance / self._max_linear_vel) // self._dt).astype(np.int)

        # p_x, p_y, v_x, v_y (in global frame), d will be added
        prediction = observation["peds_prediction"]
        # TODO: Check this
        prediction = prediction[np.arange(prediction.shape[0]), :, approx_timesteps_indices, :]
        lookahead_peds_state = np.concatenate((prediction,
                                               np.ones((prediction.shape[0], prediction.shape[1], 1)) * 100),
                                              axis=-1)
        ped_vis = observation["peds_visibility"]
        for i in range(lookahead_peds_state.shape[0]):
            lookahead_peds_state[i, ped_vis[i] > 0, :2] = lookahead_peds_state[i, ped_vis[i] > 0, :2] - subgoal_global[i]
            lookahead_peds_state[i, ped_vis[i] > 0, -1] = np.linalg.norm(
                lookahead_peds_state[i, ped_vis[i] > 0, :2], axis=-1)

        # d_goal, p_x^goal, p_y^goal, theta, v_x, v_y, omega
        goal_global = observation["goal_global"]
        if not np.allclose(robot_global[:, :2], subgoal_global):
            lookahead_theta = np.arctan2(subgoal_global[:, 1] - robot_global[:, 1],
                                         subgoal_global[:, 0] - robot_global[:, 0])
        else:
            lookahead_theta = robot_global[:, 2]
        lookahead_robot_state = np.stack([
            np.linalg.norm(subgoal_global - goal_global, axis=-1),
            subgoal_global[:, 0] - goal_global[:, 0],
            subgoal_global[:, 1] - goal_global[:, 1],
            lookahead_theta,
            self._max_linear_vel * np.cos(lookahead_theta),
            self._max_linear_vel * np.sin(lookahead_theta),
            np.zeros(lookahead_theta.shape[0])
        ], axis=-1)

        # Lookahead reward
        lookahead_reward = np.ones(lookahead_theta.shape[0])
        # TODO: Vectorize
        for i in range(lookahead_reward.shape[0]):
            if (lookahead_peds_state[i, ped_vis[i] > 0, -1] <= PEDESTRIAN_RADIUS + ROBOT_RADIUS).any():
                lookahead_reward[i] = -0.25
            elif np.linalg.norm(subgoal_global[i] - goal_global[i]) - ROBOT_RADIUS < 0.1:
                lookahead_reward[i] = 1.
            else:
                lookahead_reward[i] = -0.02

        lookahead_observation = {
            "peds": torch.Tensor(lookahead_peds_state).to(self._device),
            "peds_visibility": torch.Tensor(ped_vis).to(self._device),
            "robot": torch.Tensor(lookahead_robot_state).to(self._device)
        }
        with torch.no_grad():
            value = self._value_network(lookahead_observation)[:, 0]
            value = value.clone().detach().cpu().numpy()
        factor = np.power(0.99, self._step_cnt)
        value = lookahead_reward + factor * value  # TODO: Gamma is a parameter

        return value

    def reset(self, env_idx: Optional[int]):
        if env_idx is None:
            self._step_cnt = np.zeros(self._n_envs)
        else:
            self._step_cnt[env_idx] = 0

    def _subgoal_to_global(self, subgoal_polar: np.ndarray, robot_pose_global: np.ndarray) -> np.ndarray:
        x_rel_rot = subgoal_polar[0] * np.cos(subgoal_polar[1])
        y_rel_rot = subgoal_polar[0] * np.sin(subgoal_polar[1])
        theta = robot_pose_global[:, 2]
        x_rel = x_rel_rot * np.cos(theta) - y_rel_rot * np.sin(theta)
        y_rel = x_rel_rot * np.sin(theta) + y_rel_rot * np.cos(theta)
        x_abs = x_rel + robot_pose_global[:, 0]
        y_abs = y_rel + robot_pose_global[:, 1]
        return np.stack([x_abs, y_abs], axis=1)
