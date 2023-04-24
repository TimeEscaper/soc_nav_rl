import numpy as np
import torch
import torch.nn as nn

from abc import ABC, abstractmethod
from typing import Optional, Dict, Tuple, List

from pyminisim.core import ROBOT_RADIUS, PEDESTRIAN_RADIUS
from nip import nip

from lib.utils.math import local_polar_to_global


class AbstractSubgoalSpace(ABC):

    def __init__(self, actions: np.ndarray):
        assert len(actions.shape) == 2 and actions.shape[1] == 2, f"Actions must have shape (n_actions, 2), " \
                                                                  f"{actions.shape} is given"
        self._actions = actions
        self._n_actions = actions.shape[0]

    @property
    def actions(self) -> np.ndarray:
        return self._actions

    @property
    def n_actions(self) -> int:
        return self._n_actions

    def random_action(self) -> np.ndarray:
        idx = np.random.randint(0, self._n_actions)
        action = self._actions[idx]
        return action


@nip
class SelectiveRadialSubgoalSpace(AbstractSubgoalSpace):

    def __init__(self,
                 radial: List[int],
                 angular_n: int,
                 angular_amplitude: float):
        assert len(radial) > 0, "radial must not be empty"
        assert angular_n > 2, f"angular_n must be greater than 2, {angular_n} is given"
        assert angular_amplitude > 0, f"angular_amplitude must be greater than 0, {angular_amplitude} is given"

        if 0. not in radial:
            radial = [0.] + radial
        angular = np.linspace(-angular_amplitude, angular_amplitude, angular_n)
        if 0. not in angular:
            np.append(angular, [0.])

        actions = []
        for radial_item in radial:
            for angular_item in angular:
                actions.append([radial_item, angular_item])
        actions = np.array(actions)

        super(SelectiveRadialSubgoalSpace, self).__init__(actions)


@nip
class MaxVSubgoalPolicy:

    def __init__(self,
                 value_network: nn.Module,
                 subgoal_space: AbstractSubgoalSpace,
                 max_linear_vel: float,
                 dt: float,
                 n_envs: int,
                 rl_horizon: int,
                 device: str = "cuda"):

        self._value_network = value_network
        self._subgoal_space = subgoal_space
        self._max_linear_vel = max_linear_vel
        self._dt = dt
        self._n_envs = n_envs
        self._device = device
        self._step_cnt = {k: 0 for k in range(n_envs)}

    @property
    def action_space(self) -> AbstractSubgoalSpace:
        return self._subgoal_space

    @property
    def value_network(self) -> nn.Module:
        return self._value_network

    def predict(self, observation: Dict[str, np.ndarray], val: bool = False) -> np.ndarray:


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

    def _eval_subgoal(self,
                      subgoal: np.ndarray,
                      observation: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        robot_global = observation["robot_global"]
        goal_global = observation["goal_global"]
        subgoal_global = MaxVSubgoalPolicy.local_polar_to_global_batch(robot_global, subgoal)

        distance = np.linalg.norm(robot_global[:, :2] - subgoal_global, axis=1)
        approx_timesteps_indices = ((distance / self._max_linear_vel) // self._dt).astype(np.int)

        predictions = observation["pred_mean"][:, 1:, :]
 


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

    @staticmethod
    def local_polar_to_global_batch(robot_pose_batch: np.ndarray, point_polar: np.ndarray) -> np.ndarray:
        x_rel_rot = point_polar[0] * np.cos(point_polar[1])
        y_rel_rot = point_polar[0] * np.sin(point_polar[1])
        theta = robot_pose_batch[:, 2]
        x_rel = x_rel_rot * np.cos(theta) - y_rel_rot * np.sin(theta)
        y_rel = x_rel_rot * np.sin(theta) + y_rel_rot * np.cos(theta)
        x_abs = x_rel + robot_pose_batch[:, 0]
        y_abs = y_rel + robot_pose_batch[:, 1]
        return np.stack((x_abs, y_abs), axis=1)
