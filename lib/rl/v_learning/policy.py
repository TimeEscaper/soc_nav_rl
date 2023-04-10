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
                 device: str = "cuda"):
        self._value_network: nn.Module = None
        self._subgoal_linear = subgoal_linear.copy()
        self._subgoal_angular = subgoal_angular.copy()
        self._max_linear_vel = max_linear_vel
        self._dt = dt
        self._device = device
        self._step_cnt = 0

    @property
    def value_network(self) -> nn.Module:
        return self._value_network

    @value_network.setter
    def value_network(self, network: nn.Module):
        self._value_network = network

    def predict(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        max_value = -np.inf
        action = None
        for subgoal_linear in self._subgoal_linear:
            for subgoal_angular in self._subgoal_angular:
                action = np.array([subgoal_linear, subgoal_angular])
                value = self._eval_action(np.array([subgoal_linear, subgoal_angular]), observation)
                # print(action, value)
                if value > max_value:
                    max_value = value
                    action = np.array([subgoal_linear, subgoal_angular])
        self._step_cnt += 1
        print(f"Action: {action}, value: {max_value}")
        return action

    def _eval_action(self, subgoal_polar: np.ndarray, observation: Dict[str, np.ndarray]) -> float:
        robot_global = observation["robot_global"]
        subgoal_global = self._subgoal_to_global(subgoal_polar, observation["robot_global"])
        # print(subgoal_global)
        distance = np.linalg.norm(robot_global[:2] - subgoal_global)

        approx_timesteps_idx = (distance / self._max_linear_vel) // self._dt

        # p_x, p_y, v_x, v_y (in global frame), d will be added
        prediction = observation["peds_prediction"][:, int(approx_timesteps_idx), :]
        lookahead_peds_state = np.concatenate((prediction, np.ones((prediction.shape[0], 1)) * 100),
                                              axis=1)
        ped_vis = observation["peds_visibility"]
        lookahead_peds_state[ped_vis > 0, :2] = lookahead_peds_state[ped_vis > 0, :2] - subgoal_global

        # d_goal, p_x^goal, p_y^goal, theta, v_x, v_y, omega
        goal_global = observation["goal_global"]
        if not np.allclose(robot_global[:2], subgoal_global):
            lookahead_theta = np.arctan2(subgoal_global[1] - robot_global[1], subgoal_global[0] - robot_global[0])
        else:
            lookahead_theta = robot_global[2]
        lookahead_robot_state = np.array([
            np.linalg.norm(subgoal_global - goal_global),
            subgoal_global[0] - goal_global[0],
            subgoal_global[1] - goal_global[1],
            lookahead_theta,
            self._max_linear_vel * np.cos(lookahead_theta),
            self._max_linear_vel * np.sin(lookahead_theta),
            0.
        ])

        # Lookahead reward
        if (lookahead_peds_state[ped_vis > 0, -1] <= PEDESTRIAN_RADIUS + ROBOT_RADIUS).any():
            lookahead_reward = -0.25
        elif np.linalg.norm(subgoal_global - goal_global) - ROBOT_RADIUS < 0.1:
            lookahead_reward = 1.
        else:
            lookahead_reward = -0.02

        lookahead_observation = {
            "peds": torch.Tensor(lookahead_peds_state).unsqueeze(0).to(self._device),
            "peds_visibility": torch.Tensor(ped_vis).unsqueeze(0).to(self._device),
            "robot": torch.Tensor(lookahead_robot_state).unsqueeze(0).to(self._device)
        }
        with torch.no_grad():
            value = self._value_network(lookahead_observation)
        value = lookahead_reward + pow(0.99, self._step_cnt) * value[0].item()  # TODO: Gamma is a parameter

        return value

    def reset(self):
        self._step_cnt = 0

    def _subgoal_to_global(self, subgoal_polar: np.ndarray, robot_pose_global: np.ndarray) -> np.ndarray:
        x_rel_rot = subgoal_polar[0] * np.cos(subgoal_polar[1])
        y_rel_rot = subgoal_polar[0] * np.sin(subgoal_polar[1])
        theta = robot_pose_global[2]
        x_rel = x_rel_rot * np.cos(theta) - y_rel_rot * np.sin(theta)
        y_rel = x_rel_rot * np.sin(theta) + y_rel_rot * np.cos(theta)
        x_abs = x_rel + robot_pose_global[0]
        y_abs = y_rel + robot_pose_global[1]
        return np.array([x_abs, y_abs])
