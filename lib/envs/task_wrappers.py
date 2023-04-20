import gym
import numpy as np

from abc import ABC, abstractmethod
from typing import Union, List, Any, Dict, Tuple, Optional, Callable
from pyminisim.core import SimulationState, ROBOT_RADIUS, PEDESTRIAN_RADIUS
from pyminisim.util import wrap_angle
from pyminisim.visual import AbstractDrawing, CircleDrawing, Covariance2dDrawing
from nip import nip

from lib.controllers import AbstractControllerFactory
from lib.envs.sim_config_samplers import ProblemConfig, SimConfig
from lib.envs.curriculum import AbstractCurriculum
from lib.envs.core_env import PyMiniSimCoreEnv
from lib.predictors import PedestrianTracker
from lib.utils.math import unnormalize_symmetric, local_polar_to_global


class AbstractTaskEnv(ABC, gym.Env):

    @abstractmethod
    def step(self, action) -> \
            Tuple[Optional[Union[np.ndarray, Dict[str, np.ndarray]]], Optional[float], bool, Dict[str, Any]]:
        raise NotImplementedError()

    @abstractmethod
    def reset(self) -> Optional[Union[np.ndarray, Dict[str, np.ndarray]]]:
        raise NotImplementedError()

    @abstractmethod
    def update_curriculum(self):
        raise NotImplementedError()

    @abstractmethod
    def get_simulation_state(self) -> SimulationState:
        raise NotImplementedError()

    @abstractmethod
    def get_problem_config(self) -> ProblemConfig:
        raise NotImplementedError()

    @abstractmethod
    def get_goal(self) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def enable_render(self):
        raise NotImplementedError()

    @abstractmethod
    def draw(self, name: str, drawing: AbstractDrawing):
        raise NotImplementedError()

    @abstractmethod
    def clear_drawing(self, name: Union[str, List[str]]):
        raise NotImplementedError()

    def render(self, mode="human"):
        pass


class AbstractTaskWrapper(AbstractTaskEnv):

    def __init__(self, env: AbstractTaskEnv):
        super(AbstractTaskWrapper, self).__init__()
        self._env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.reward_range = env.reward_range

    def step(self, action) -> \
            Tuple[Optional[Union[np.ndarray, Dict[str, np.ndarray]]], Optional[float], bool, Dict[str, Any]]:
        return self._env.step(action)

    def reset(self) -> Optional[Union[np.ndarray, Dict[str, np.ndarray]]]:
        return self._env.reset()

    def update_curriculum(self):
        self._env.update_curriculum()

    def get_simulation_state(self) -> SimulationState:
        return self._env.get_simulation_state()

    def get_problem_config(self) -> ProblemConfig:
        return self._env.get_problem_config()

    def get_goal(self) -> np.ndarray:
        return self._env.get_goal()

    def enable_render(self):
        return self._env.enable_render()

    def draw(self, name: str, drawing: AbstractDrawing):
        self._env.draw(name, drawing)

    def clear_drawing(self, name: Union[str, List[str]]):
        self._env.clear_drawing(name)


class BaseEnv(AbstractTaskEnv):

    def __init__(self, sim_config: SimConfig, curriculum: AbstractCurriculum, is_eval: bool):
        super(BaseEnv, self).__init__()
        self._env = PyMiniSimCoreEnv(sim_config=sim_config, curriculum=curriculum, is_eval=is_eval)

        self.observation_space = None
        self.action_space = None

    def step(self, action: np.ndarray) -> \
            Tuple[Optional[Union[np.ndarray, Dict[str, np.ndarray]]], Optional[float], bool, Dict[str, Any]]:
        has_collision, goal_reached, min_separation_distance = self._env.step(action)
        done = has_collision or goal_reached
        info = {"min_separation": min_separation_distance}
        if has_collision:
            info["done_reason"] = "collision"
            info["is_success"] = False
        elif goal_reached:
            info["done_reason"] = "success"
            info["is_success"] = True
        return None, None, done, info

    def reset(self) -> Optional[Union[np.ndarray, Dict[str, np.ndarray]]]:
        self._env.reset()
        return None

    def update_curriculum(self):
        self._env.update_curriculum()

    def get_simulation_state(self) -> SimulationState:
        return self._env.sim_state

    def get_problem_config(self) -> ProblemConfig:
        return self._env.problem_config

    def get_goal(self) -> np.ndarray:
        return self._env.goal

    def enable_render(self):
        self._env.enable_render()

    def draw(self, name: str, drawing: AbstractDrawing):
        self._env.draw(name, drawing)

    def clear_drawing(self, name: Union[str, List[str]]):
        self._env.clear_drawing(name)


@nip
class UnicycleEnv(AbstractTaskWrapper):

    def __init__(self,
                 env: AbstractTaskEnv,
                 lb: Tuple[float, float],
                 ub: Tuple[float, float],
                 normalize: bool = False,
                 dtype=np.float32):
        super(UnicycleEnv, self).__init__(env)
        assert len(lb) == 2 and len(lb) == len(ub), f"lb and ub must have length 2, got {len(lb)} and {len(ub)}"
        assert ub[0] > lb[0] and ub[1] > lb[1], f"ub must be greater then lb everywhere, got {lb} and {ub}"

        self._lb = np.array(lb)
        self._ub = np.array(ub)
        self._normalize = normalize

        if self._normalize:
            self.action_space = gym.spaces.Box(low=np.array([-1., -1.]),
                                               high=np.array([1., 1.]),
                                               shape=(2,),
                                               dtype=dtype)
        else:
            self.action_space = gym.spaces.Box(low=self._lb.copy(),
                                               high=self._ub.copy(),
                                               shape=(2,),
                                               dtype=dtype)

    def step(self, action: np.ndarray) -> \
            Tuple[Optional[Union[np.ndarray, Dict[str, np.ndarray]]], Optional[float], bool, Dict[str, Any]]:
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self._normalize:
            action = unnormalize_symmetric(action, self._lb, self._ub)
        return self._env.step(action)


@nip
class SARLObservationEnv(AbstractTaskWrapper):

    def __init__(self,
                 env: AbstractTaskEnv,
                 peds_padding: int,
                 dtype=np.float32):
        super(SARLObservationEnv, self).__init__(env)
        self._peds_padding = peds_padding

        if isinstance(env.observation_space, gym.spaces.Dict):
            obs_dict = env.observation_space.spaces.copy()
        else:
            obs_dict = {}
        obs_dict.update({
            # d_g, phi_goal, v_x, v_y, r
            "robot": gym.spaces.Box(low=np.array([0., -np.pi, -np.inf, -np.inf, 0.]),
                                    high=np.array([np.inf, np.pi, np.inf, np.inf, ROBOT_RADIUS]),
                                    shape=(5,),
                                    dtype=dtype),
            # p_x, p_y, phi, v_x, v_y, r_ped, d, r_ped + r_robot
            "peds": gym.spaces.Box(low=np.tile(np.array([-np.inf, -np.inf, -np.pi, -np.inf, -np.inf, 0., 0., 0.]),
                                               (peds_padding, 1)),
                                   high=np.tile(np.array([np.inf, np.inf, np.pi, np.inf, np.inf, PEDESTRIAN_RADIUS,
                                                          np.inf, PEDESTRIAN_RADIUS + ROBOT_RADIUS]),
                                                (peds_padding, 1)),
                                   shape=(peds_padding, 8),
                                   dtype=dtype),
            "visibility": gym.spaces.Box(low=False,
                                         high=True,
                                         shape=(peds_padding,),
                                         dtype=np.bool)
        })
        self.observation_space = gym.spaces.Dict(obs_dict)

    def step(self, action: np.ndarray):
        obs, reward, done, info = self._env.step(action)
        if not isinstance(obs, dict):
            obs = {}
        obs.update(self._build_observation())
        return obs, reward, done, info

    def reset(self):
        obs_dict = self._env.reset()
        if not isinstance(obs_dict, dict):
            obs_dict = {}
        obs_dict.update(self._build_observation())
        return obs_dict

    def _build_observation(self):
        sim_state: SimulationState = self._env.get_simulation_state()
        goal: np.ndarray = self._env.get_goal()

        robot_pose = sim_state.world.robot.pose
        robot_vel = sim_state.world.robot.velocity
        rotation_matrix = np.array([[np.cos(robot_pose[2]), -np.sin(robot_pose[2])],
                                    [np.sin(robot_pose[2]), np.cos(robot_pose[2])]])
        robot_vel = rotation_matrix @ robot_vel[:2]

        d_g = np.linalg.norm(robot_pose[:2] - goal)
        phi_goal = wrap_angle(robot_pose[2] - np.arctan2(goal[1] - robot_pose[1], goal[0] - robot_pose[0]))
        robot_obs = np.array([d_g, phi_goal, robot_vel[0], robot_vel[1], ROBOT_RADIUS])

        peds_obs = np.tile(np.array([-10., -10., 0., 0., 0., 0., 100, ROBOT_RADIUS]), (self._peds_padding, 1))
        peds_vis = np.zeros(self._peds_padding, dtype=np.bool)
        detections = self.get_simulation_state().sensors["pedestrian_detector"].reading.pedestrians
        for i, (k, v) in enumerate(detections.items()):
            ped_pose = np.array(v)
            ped_pose = ped_pose - robot_pose[:2]
            ped_pose = rotation_matrix @ ped_pose
            ped_phi = np.arctan2(ped_pose[1], ped_pose[0])
            ped_vel = rotation_matrix @ sim_state.world.pedestrians.velocities[k][:2]
            d = np.linalg.norm(ped_pose)
            peds_obs[i] = np.array([ped_pose[0], ped_pose[1], ped_phi, ped_vel[0], ped_vel[1], PEDESTRIAN_RADIUS,
                                    d, PEDESTRIAN_RADIUS + ROBOT_RADIUS])
            peds_vis[i] = True

        return {
            "robot": robot_obs,
            "peds": peds_obs,
            "visibility": peds_vis
        }


@nip
class SARLRewardEnv(AbstractTaskWrapper):

    def __init__(self,
                 env: AbstractTaskEnv,
                 success_reward: float = 1.,
                 collision_reward: float = -0.25,
                 separation_reward_offset: float = -0.1,
                 separation_threshold: float = 0.2,
                 step_reward: float = 0.):
        super(SARLRewardEnv, self).__init__(env)
        self._success_reward = success_reward
        self._collision_reward = collision_reward
        self._separation_reward_offset = separation_reward_offset
        self._separation_threshold = separation_threshold
        self._step_reward = step_reward

        self.reward_range = (collision_reward, success_reward)

    def step(self, action: np.ndarray):
        obs, _, done, info = self._env.step(action)

        if done:
            if "done_reason" not in info:
                raise ValueError("done_reason must be contained in info when done")
            done_reason = info["done_reason"]
            if done_reason == "collision":
                reward = self._collision_reward
            elif done_reason == "success":
                reward = self._success_reward
            else:
                reward = self._step_reward
        else:
            if "min_separation" in info:
                min_separation = info["min_separation"]
                if min_separation < self._separation_threshold:
                    reward = self._separation_reward_offset + min_separation / 2.
                else:
                    reward = self._step_reward
            else:
                reward = self._step_reward

        return obs, reward, done, info


@nip
class SARLPredictionRewardEnv(AbstractTaskWrapper):

    def __init__(self,
                 env: AbstractTaskEnv,
                 success_reward: float = 1.,
                 collision_reward: float = -0.25,
                 separation_reward_factor: float = -0.1,
                 separation_threshold: float = 0.2,
                 step_reward: float = -0.01):
        super(SARLPredictionRewardEnv, self).__init__(env)
        self._success_reward = success_reward
        self._collision_reward = collision_reward
        self._separation_reward_factor = separation_reward_factor
        self._separation_threshold = separation_threshold
        self._step_reward = step_reward

        self.reward_range = (collision_reward, success_reward)
        self._previous_obs = None

    def step(self, action: np.ndarray):
        obs, _, done, info = self._env.step(action)

        if done:
            if "done_reason" not in info:
                raise ValueError("done_reason must be contained in info when done")
            done_reason = info["done_reason"]
            if done_reason == "collision":
                reward = self._collision_reward
            elif done_reason == "success":
                reward = self._success_reward
            else:
                reward = self._step_reward
        else:
            pred_means = self._previous_obs["pred_mean"]
            pred_vis = self._previous_obs["visibility"]
            robot_pose = self._env.get_simulation_state().world.robot.pose[:2]
            reward = 0.
            for i in range(pred_means.shape[0]):
                if not pred_vis[i]:
                    continue
                distances = np.linalg.norm(robot_pose - pred_means[i, :, :], axis=1)
                min_distance = np.min(distances)
                min_distance_idx = np.argmin(distances)
                if min_distance - ROBOT_RADIUS - PEDESTRIAN_RADIUS >= self._separation_threshold:
                    continue
                separation_reward = self._separation_reward_factor / (min_distance_idx + 1)
                if separation_reward < reward:
                    reward = separation_reward

        self._previous_obs = obs
        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        self._previous_obs = obs
        return obs


@nip
class TimeLimitEnv(AbstractTaskWrapper):

    def __init__(self,
                 env: AbstractTaskEnv):
        super(TimeLimitEnv, self).__init__(env)
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self._step_cnt = 0

    def step(self, action: np.ndarray):
        obs, reward, done, info = self._env.step(action)
        self._step_cnt += 1

        if not done and self._step_cnt >= self._env.get_problem_config().max_steps:
            done = True
            info["TimeLimit.truncated"] = True
            info["done_reason"] = "truncated"
            info["is_success"] = False
        # TODO: Should we have this else branch?
        # else:
        #     info["TimeLimit.truncated"] = False

        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        self._step_cnt = 0
        return obs


@nip
class PredictionEnv(AbstractTaskWrapper):

    def __init__(self,
                 env: AbstractTaskEnv,
                 tracker_factory: Callable[..., PedestrianTracker],
                 peds_padding: int,
                 dtype=np.float32):
        super(PredictionEnv, self).__init__(env)
        self._tracker = tracker_factory()
        self._peds_padding = peds_padding

        if isinstance(env.observation_space, gym.spaces.Dict):
            obs_dict = self.observation_space.spaces.copy()
        else:
            obs_dict = {}
        obs_dict.update({
            "pred_mean": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(peds_padding, self._tracker.horizon + 1, 2),  # Current pose + prediction = horizon + 1
                dtype=dtype
            ),
            "pred_cov": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(peds_padding, self._tracker.horizon, 2, 2),  # Covariances only for prediction
                dtype=dtype
            ),
            "visibility": gym.spaces.Box(low=False,
                                         high=True,
                                         shape=(peds_padding,),
                                         dtype=np.bool)
        })
        self.observation_space = gym.spaces.Dict(obs_dict)

    def step(self, action: np.ndarray):
        obs, reward, done, info = self._env.step(action)
        if not isinstance(obs, dict):
            obs = {}

        self._update_tracker()

        obs.update(self._build_observation())
        self._env.draw("pred_mean", CircleDrawing(obs["pred_mean"][:, 1:, :].reshape(-1, 2),
                                                  0.05, (173, 153, 121)))
        self._env.draw("pred_cov", Covariance2dDrawing(obs["pred_mean"][:, 1:, :].reshape(-1, 2),
                                                       obs["pred_cov"].reshape(-1, 2, 2),
                                                       (173, 153, 121), 0.05, n_sigma=1))

        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        if not isinstance(obs, dict):
            obs = {}

        self._tracker.reset()
        self._update_tracker()

        obs.update(self._build_observation())
        return obs

    def _update_tracker(self):
        sim_state = self._env.get_simulation_state()
        detections = {k: np.array([v[0], v[1],
                                   sim_state.world.pedestrians.velocities[k][0],
                                   sim_state.world.pedestrians.velocities[k][1]])
                      for k, v in sim_state.sensors["pedestrian_detector"].reading.pedestrians.items()}
        self._tracker.update(detections)

    def _build_observation(self) -> Dict[str, np.ndarray]:

        current_poses = self._tracker.get_current_poses()
        obs_pred_mean = np.tile(np.array([-10., -10.]), (self._peds_padding, self._tracker.horizon + 1, 1))
        obs_pred_cov = np.tile(np.eye(2) * 0.001, (self._peds_padding, self._tracker.horizon, 1, 1))
        peds_vis = np.zeros(self._peds_padding, dtype=np.bool)

        for i, (k, v) in enumerate(self._tracker.get_predictions().items()):
            obs_pred_mean[i, :, :] = np.concatenate((current_poses[k][np.newaxis], v[0]), axis=0)
            obs_pred_cov[i] = v[1]
            peds_vis[i] = True

        return {
            "pred_mean": obs_pred_mean,
            "pred_cov": obs_pred_cov,
            "visibility": peds_vis
        }


@nip
class SubgoalEnv(AbstractTaskWrapper):

    def __init__(self,
                 env: AbstractTaskEnv,
                 controller_factory: AbstractControllerFactory,
                 lb: Tuple[float, float] = (0.7, -np.deg2rad(110.)),
                 ub: Tuple[float, float] = (3., np.deg2rad(110)),
                 normalize: bool = False,
                 subgoal_reach_threshold: float = 0.1,
                 max_subgoal_steps: Optional[int] = None,
                 dtype=np.float32):
        super(SubgoalEnv, self).__init__(env)
        self._controller = controller_factory()

        self._lb = np.array(lb)
        self._ub = np.array(ub)
        self._normalize = normalize
        self._subgoal_reach_threshold = subgoal_reach_threshold
        self._max_steps = max_subgoal_steps if max_subgoal_steps is not None else self._controller.horizon

        if normalize:
            self.action_space = gym.spaces.Box(
                low=-np.ones(2, dtype=dtype),
                high=np.ones(2, dtype=dtype),
                shape=(2,),
                dtype=dtype
            )
        else:
            self.action_space = gym.spaces.Box(
                low=self._lb.copy(),
                high=self._ub.copy(),
                shape=(2,),
                dtype=dtype
            )

        self._pred_mean: Optional[np.ndarray] = None
        self._pred_cov: Optional[np.ndarray] = None
        self._pred_vis: Optional[np.ndarray] = None

    def step(self, action: np.ndarray):
        if self._normalize:
            action = unnormalize_symmetric(action, self._lb, self._ub)
        robot_pose = self._env.get_simulation_state().world.robot.pose
        subgoal = local_polar_to_global(robot_pose, action)
        self._env.draw("subgoal", CircleDrawing(subgoal, 0.05, (0, 0, 255)))

        self._controller.set_goal(state=robot_pose, goal=subgoal)
        step_cnt = 0
        obs = None
        reward = None
        done = None
        info = None
        while step_cnt < self._max_steps:
            if self._pred_vis.any():
                predictions = (self._pred_mean[self._pred_vis, :, :], self._pred_cov[self._pred_vis, :, :, :])
            else:
                predictions = None
            control, control_info = self._controller.step(robot_pose, predictions)
            if "mpc_traj" in control_info:
                self._env.draw(f"mpc_traj", CircleDrawing(control_info["mpc_traj"], 0.04, (209, 133, 128)))

            obs, reward, done, info = self._env.step(control)
            step_cnt += 1
            self._pred_mean = obs["pred_mean"][:, 1:, :]
            self._pred_cov = obs["pred_cov"]
            self._pred_vis = obs["visibility"]

            robot_pose = self._env.get_simulation_state().world.robot.pose
            if done or np.linalg.norm(robot_pose[:2] - subgoal) - ROBOT_RADIUS < self._subgoal_reach_threshold:
                break

        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        self._pred_mean = obs["pred_mean"][:, 1:, :]
        self._pred_cov = obs["pred_cov"]
        self._pred_vis = obs["visibility"]
        return obs


@nip
class SARLPredictionEnv(AbstractTaskWrapper):

    def __init__(self,
                 env: AbstractTaskEnv,
                 rl_horizon: int,
                 peds_padding: int,
                 overwrite_obs: bool = True,
                 dtype=np.float32):
        super(SARLPredictionEnv, self).__init__(env)
        self._rl_horizon = rl_horizon
        self._overwrite_obs = overwrite_obs

        if isinstance(env.observation_space, gym.spaces.Dict) and not overwrite_obs:
            obs_dict = self.observation_space.spaces.copy()
        else:
            obs_dict = {}
        obs_dict.update({
            "robot": gym.spaces.Box(low=np.array([0., -np.pi, -np.inf, -np.inf, 0.]),
                                    high=np.array([np.inf, np.pi, np.inf, np.inf, ROBOT_RADIUS]),
                                    shape=(5,),
                                    dtype=dtype),
            "pred_mean_rl": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(peds_padding, rl_horizon + 1, 2),  # Current pose + prediction = horizon + 1
                dtype=dtype
            ),
            "pred_cov_rl": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(peds_padding, rl_horizon, 2, 2),  # Covariances only for prediction
                dtype=dtype
            ),
            "visibility": gym.spaces.Box(low=False,
                                         high=True,
                                         shape=(peds_padding,),
                                         dtype=np.bool)
        })
        self.observation_space = gym.spaces.Dict(obs_dict)

    def step(self, action: np.ndarray):
        obs, reward, done, info = self._env.step(action)
        obs_new = self._rebuild_observation(obs)
        if self._overwrite_obs:
            obs = obs_new
        else:
            obs.update(obs_new)
        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        obs_new = self._rebuild_observation(obs)
        if self._overwrite_obs:
            obs = obs_new
        else:
            obs.update(obs_new)
        return obs

    def _rebuild_observation(self, obs_original: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        robot_pose = self._env.get_simulation_state().world.robot.pose
        robot_vel = self._env.get_simulation_state().world.robot.velocity
        goal = self._env.get_goal()

        rotation_matrix = np.array([[np.cos(robot_pose[2]), -np.sin(robot_pose[2])],
                                    [np.sin(robot_pose[2]), np.cos(robot_pose[2])]])
        robot_vel = rotation_matrix @ robot_vel[:2]
        d_g = np.linalg.norm(robot_pose[:2] - goal)
        phi_goal = wrap_angle(robot_pose[2] - np.arctan2(goal[1] - robot_pose[1], goal[0] - robot_pose[0]))
        obs_robot = np.array([d_g, phi_goal, robot_vel[0], robot_vel[1], ROBOT_RADIUS])

        pred_mean = obs_original["pred_mean"][:, :self._rl_horizon + 1, :] - robot_pose[:2]
        pred_cov = obs_original["pred_cov"][:, :self._rl_horizon, :, :]
        pred_vis = obs_original["visibility"]

        if pred_vis.any():
            pred_mean[pred_vis] = np.einsum("ij,mnj->mni", rotation_matrix, pred_mean[pred_vis])
            pred_cov[pred_vis] = np.einsum("ij,mnjk->mnik", rotation_matrix, pred_cov[pred_vis])
            pred_cov[pred_vis] = np.einsum("mnij,jk->mnik", pred_cov[pred_vis], rotation_matrix.T)

        return {
            "robot": obs_robot,
            "pred_mean_rl": pred_mean,
            "pred_cov_rl": pred_cov,
            "visibility": pred_vis
        }


@nip
class EnvWrapEntry:

    def __init__(self, env_cls: Callable[..., AbstractTaskEnv], kwargs: Optional[Dict[str, Any]] = None):
        self._env_cls = env_cls
        self._kwargs = kwargs if kwargs is not None else {}

    @property
    def env_cls(self) -> Callable[..., AbstractTaskEnv]:
        return self._env_cls

    @property
    def kwargs(self) -> Dict[str, Any]:
        return self._kwargs


@nip
class WrappedEnvFactory:

    def __init__(self,
                 sim_config: SimConfig,
                 curriculum: AbstractCurriculum,
                 wrappers: List[EnvWrapEntry]):
        self._sim_config = sim_config
        self._curriculum = curriculum
        self._wrappers = wrappers

    def __call__(self, is_eval: bool) -> gym.Env:
        env = BaseEnv(sim_config=self._sim_config,
                      curriculum=self._curriculum,
                      is_eval=is_eval)
        for wrapper in self._wrappers:
            env = wrapper.env_cls(env, **wrapper.kwargs)
        return env
