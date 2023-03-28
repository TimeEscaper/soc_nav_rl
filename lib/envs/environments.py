from __future__ import annotations

import random
import numpy as np
import gym

from abc import ABC, abstractmethod
from typing import Dict, Union, Optional, Tuple, List, Any, Callable
from dataclasses import dataclass
from nip import nip
from scipy.spatial.distance import cdist
from lib.envs.rewards import AbstractReward, RewardContext
from lib.envs.agents import AbstractAgentsSampler
from lib.predictors.tracker import PedestrianTracker

from pyminisim.core import PEDESTRIAN_RADIUS, ROBOT_RADIUS
from pyminisim.core import Simulation, SimulationState
from pyminisim.world_map import EmptyWorld, CirclesWorld
from pyminisim.robot import UnicycleRobotModel
from pyminisim.pedestrians import HeadedSocialForceModelPolicy, OptimalReciprocalCollisionAvoidance, \
    RandomWaypointTracker, FixedWaypointTracker
from pyminisim.sensors import PedestrianDetectorNoise, PedestrianDetector, PedestrianDetectorConfig, \
    LidarSensor, LidarSensorNoise
from pyminisim.visual import Renderer, CircleDrawing, AbstractDrawing


def get_or_sample_uniform(value: Optional[Union[Any, Tuple[Any, Any]]], size: Optional[Tuple[int, ...]] = None) -> Any:
    if value is None:
        return None
    if isinstance(value, tuple):
        return np.random.uniform(value[0], value[1], size=size)
    return value


def get_or_sample_choice(value: Optional[Union[Any, Tuple[Any, ...], List[Any, ...]]]) -> Any:
    if value is None:
        return None
    if isinstance(value, tuple) or isinstance(value, list):
        return random.choice(value)
    return value


def get_or_sample_bool(value: Optional[Union[str, bool]]) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, str):
        return random.choice([True, False])
    return value


@dataclass
@nip
class SimConfig:
    ped_model: Optional[Union[str, Tuple[str, ...]]]
    robot_visible: Union[bool, str] = False
    detector_range: Optional[Union[float, Tuple[float, float]]] = 5
    detector_fov: Union[float, Tuple[float, float]] = 360.
    control_lb: Tuple[float, float] = (0., 0.)
    control_ub: Tuple[float, float] = (2., 0.9 * np.pi)
    goal_reach_threshold: float = 0.1
    max_steps: int = 300
    sim_dt: float = 0.01
    policy_dt: float = 0.1
    rt_factor: Optional[float] = None

    def sample(self) -> SimConfig:
        return SimConfig(ped_model=get_or_sample_choice(self.ped_model),
                         robot_visible=get_or_sample_bool(self.robot_visible),
                         detector_range=get_or_sample_uniform(self.detector_range),
                         detector_fov=get_or_sample_uniform(self.detector_fov),
                         control_lb=self.control_lb,
                         control_ub=self.control_ub,
                         goal_reach_threshold=self.goal_reach_threshold,
                         max_steps=self.max_steps,
                         sim_dt=self.sim_dt,
                         policy_dt=self.policy_dt,
                         rt_factor=self.rt_factor)

    def is_deterministic(self) -> bool:
        randomized = isinstance(self.ped_model, tuple) and \
                     isinstance(self.robot_visible, str) and \
                     isinstance(self.detector_range, tuple) and \
                     isinstance(self.detector_fov, tuple)
        return not randomized


class AbstractEnvFactory(ABC):

    @abstractmethod
    def __call__(self) -> gym.Env:
        raise NotImplementedError()


class PyMiniSimWrap:

    def __init__(self,
                 agents_sampler: AbstractAgentsSampler,
                 sim_config: SimConfig,
                 render: bool = False,
                 normalize_actions: bool = False):
        self._config_sampler = sim_config
        self._agents_sampler = agents_sampler
        self._render = render

        self._step_cnt = 0

        self._sim: Simulation = None
        self._renderer: Renderer = None
        self._robot_goal: np.ndarray = None
        self._config: SimConfig = None

        self._has_peds = sim_config.ped_model != "none"

        if not normalize_actions:
            self.action_space = gym.spaces.Box(
                low=np.array(sim_config.control_lb),
                high=np.array(sim_config.control_ub),
                shape=(2,),
                dtype=np.float32
            )
        else:
            self.action_space = gym.spaces.Box(
                low=np.array([-1., -1.]),
                high=np.array([1., 1.]),
                shape=(2,),
                dtype=np.float32
            )
        self._normalize_actions = normalize_actions

    @property
    def goal(self) -> np.ndarray:
        return self._robot_goal

    @property
    def current_step_cnt(self) -> int:
        return self._step_cnt

    @property
    def has_pedestrians(self) -> bool:
        return self._has_peds

    @property
    def render_enabled(self) -> bool:
        return self._render

    @property
    def current_config(self) -> SimConfig:
        return self._config

    @property
    def sim_state(self) -> SimulationState:
        return self._sim.current_state

    def step(self, action: np.ndarray) -> Tuple[bool, bool, bool]:
        action = np.clip(action,
                         self.action_space.low, self.action_space.high)
        if self._normalize_actions:
            deviation = (np.array(self._config.control_ub) - np.array(self._config.control_lb)) / 2.
            shift = (np.array(self._config.control_ub) + np.array(self._config.control_lb)) / 2.
            action = (action * deviation) + shift

        hold_time = 0.
        has_collision = False
        while hold_time < self._config.policy_dt:
            self._sim.step(action)
            hold_time += self._config.sim_dt
            if self._renderer is not None:
                self._renderer.render()
            collisions = self._sim.current_state.world.robot_to_pedestrians_collisions
            has_collision = collisions is not None and len(collisions) > 0
            if has_collision:
                break

        self._step_cnt += 1
        truncated = (self._step_cnt > self._config.max_steps) and not has_collision

        if has_collision or truncated:
            success = False
        else:
            success = np.linalg.norm(
                self._sim.current_state.world.robot.pose[:2] - self._robot_goal) \
                      - ROBOT_RADIUS < self._config.goal_reach_threshold

        return has_collision, truncated, success

    def reset(self):
        config = self._config_sampler.sample()
        self._config = config

        agents_sample = self._agents_sampler.sample()
        self._robot_goal = agents_sample.robot_goal

        robot_model = UnicycleRobotModel(initial_pose=agents_sample.robot_initial_pose,
                                         initial_control=np.array([0.0, np.deg2rad(0.0)]))

        if self._has_peds:
            if agents_sample.ped_goals is None:
                waypoint_tracker = RandomWaypointTracker(world_size=agents_sample.world_size)
            else:
                waypoint_tracker = FixedWaypointTracker(waypoints=agents_sample.ped_goals[np.newaxis, :, :])

            if config.ped_model == "hsfm":
                ped_model = HeadedSocialForceModelPolicy(waypoint_tracker=waypoint_tracker,
                                                         n_pedestrians=agents_sample.n_peds,
                                                         initial_poses=agents_sample.ped_initial_poses,
                                                         robot_visible=config.robot_visible)
            elif config.ped_model == "orca":
                ped_model = OptimalReciprocalCollisionAvoidance(dt=config.sim_dt,
                                                                waypoint_tracker=waypoint_tracker,
                                                                n_pedestrians=agents_sample.n_peds,
                                                                initial_poses=agents_sample.ped_initial_poses,
                                                                robot_visible=config.robot_visible)
            else:
                raise ValueError()
        else:
            ped_model = None

        ped_detector = PedestrianDetector(
            config=PedestrianDetectorConfig(max_dist=config.detector_range,
                                            fov=config.detector_fov,
                                            return_type=PedestrianDetectorConfig.RETURN_ABSOLUTE))

        sim = Simulation(world_map=EmptyWorld(),
                         robot_model=robot_model,
                         pedestrians_model=ped_model,
                         sensors=[ped_detector],
                         sim_dt=config.sim_dt,
                         rt_factor=config.rt_factor)
        if self._render:
            renderer = Renderer(simulation=sim,
                                resolution=50.,
                                screen_size=(1000, 1000))
            renderer.draw("goal", CircleDrawing(center=self._robot_goal[:2],
                                                radius=0.05,
                                                color=(255, 0, 0)))
        else:
            renderer = None

        self._sim = sim
        self._renderer = renderer

        self._step_cnt = 0

    def draw(self, drawing_id: str, drawing: AbstractDrawing):
        if self._renderer is not None:
            self._renderer.draw(drawing_id, drawing)

    def clear_drawing(self, drawing_id: str):
        if self._renderer is not None:
            self._renderer.clear_drawings([drawing_id])

    def enable_render(self):
        self._render = True


@nip
class SimpleNavEnv(gym.Env):

    def __init__(self,
                 agents_sampler: AbstractAgentsSampler,
                 reward: AbstractReward,
                 sim_config: SimConfig,
                 render: bool = False,
                 normalize_actions: bool = False):
        assert sim_config.ped_model == "none", "Pedestrians are not supported for the simple navigation env"
        self._sim_wrap = PyMiniSimWrap(agents_sampler,
                                       sim_config,
                                       render,
                                       normalize_actions)
        self._reward = reward

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4,),
            dtype=np.float32
        )
        self.action_space = self._sim_wrap.action_space

    def step(self, action: np.ndarray):
        previous_robot_pose = self._sim_wrap.sim_state.world.robot.pose
        goal = self._sim_wrap.goal

        collision, truncated, success = self._sim_wrap.step(action)
        robot_pose = self._sim_wrap.sim_state.world.robot.pose

        reward_context = RewardContext()
        reward_context.set("goal", goal)
        reward_context.set("robot_pose", robot_pose)
        reward_context.set("previous_robot_pose", previous_robot_pose)

        if collision:
            done = True
            info = {"done_reason": "collision"}
            reward_context.set("collision", True)
        elif truncated:
            done = True
            info = {"done_reason": "truncated",
                    "TimeLimit.truncated": True}  # https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html#tips-and-tricks-when-creating-a-custom-environment
            reward_context.set("truncated", True)
        elif success:
            done = True
            info = {"done_reason": "success"}
            reward_context.set("success", True)
        else:
            done = False
            info = {}

        reward = self._reward(reward_context)

        observation = SimpleNavEnv._build_observation(robot_pose, goal)

        return observation, reward, done, info

    def reset(self):
        self._sim_wrap.reset()
        goal = self._sim_wrap.goal
        robot_pose = self._sim_wrap.sim_state.world.robot.pose
        observation = SimpleNavEnv._build_observation(robot_pose, goal)
        return observation

    def render(self, mode="human"):
        pass

    def enable_render(self):
        self._sim_wrap.enable_render()

    @staticmethod
    def _build_observation(robot_pose: np.ndarray, goal: np.ndarray) -> np.ndarray:
        return np.array([np.linalg.norm(goal[:2] - robot_pose[:2]),
                         goal[0] - robot_pose[0],
                         goal[1] - robot_pose[1],
                         robot_pose[2]]).astype(np.float32)


@nip
class SimpleNavEnvFactory(AbstractEnvFactory):

    def __init__(self,
                 agents_sampler: AbstractAgentsSampler,
                 reward: AbstractReward,
                 sim_config: SimConfig,
                 render: bool = False,
                 normalize_actions: bool = False):
        self._agents_sampler = agents_sampler
        self._reward = reward
        self._sim_config = sim_config
        self._render = render
        self._normalize_actions = normalize_actions

    def __call__(self) -> SimpleNavEnv:
        return SimpleNavEnv(self._agents_sampler,
                            self._reward,
                            self._sim_config,
                            self._render,
                            self._normalize_actions)


@nip
class SocialNavGraphEnv(gym.Env):

    def __init__(self,
                 agents_sampler: AbstractAgentsSampler,
                 ped_tracker: PedestrianTracker,
                 reward: AbstractReward,
                 sim_config: SimConfig,
                 render: bool = False,
                 normalize_actions: bool = False):
        self._sim_wrap = PyMiniSimWrap(agents_sampler,
                                       sim_config,
                                       render,
                                       normalize_actions)
        self._reward = reward
        self._ped_tracker = ped_tracker

        self._max_peds = agents_sampler.max_peds

        self._previous_ped_predictions = ped_tracker.get_predictions()

        self.observation_space = gym.spaces.Dict({
            "peds_traj": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self._max_peds, ped_tracker.horizon + 1, 2),  # Current state + predictions = 1 + horizon
                dtype=np.float
            ),
            "peds_visibility": gym.spaces.Box(
                low=False,
                high=True,
                shape=(self._max_peds,),
                dtype=np.bool
            ),
            "robot_state": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(4,),
                dtype=np.float
            )
        })

        self.action_space = self._sim_wrap.action_space

    def step(self, action: np.ndarray):
        previous_robot_pose = self._sim_wrap.sim_state.world.robot.pose
        previous_predictions = self._ped_tracker.get_predictions()
        goal = self._sim_wrap.goal

        collision, truncated, success = self._sim_wrap.step(action)
        self._ped_tracker.update(self._get_detections())
        robot_pose = self._sim_wrap.sim_state.world.robot.pose

        if self._sim_wrap.render_enabled:
            current_predictions = self._ped_tracker.get_predictions()
            for k, pred in current_predictions.items():
                self._sim_wrap.draw(f"pred_{k}", CircleDrawing(pred[0], 0.05, (0, 0, 255)))
            for k in set(previous_predictions.keys()).difference(set(current_predictions.keys())):
                self._sim_wrap.clear_drawing(f"pred_{k}")

        reward_context = RewardContext()
        reward_context.set("goal", goal)
        reward_context.set("robot_pose", robot_pose)
        reward_context.set("previous_robot_pose", previous_robot_pose)
        reward_context.set("previous_ped_predictions", previous_predictions)

        if collision:
            done = True
            info = {"done_reason": "collision"}
            reward_context.set("collision", True)
        elif truncated:
            done = True
            info = {"done_reason": "truncated",
                    "TimeLimit.truncated": True}  # https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html#tips-and-tricks-when-creating-a-custom-environment
            reward_context.set("truncated", True)
        elif success:
            done = True
            info = {"done_reason": "success"}
            reward_context.set("success", True)
        else:
            done = False
            info = {}

        reward = self._reward(reward_context)

        observation = self._build_obs()

        return observation, reward, done, info

    def reset(self):
        self._sim_wrap.reset()
        self._ped_tracker.reset()
        self._ped_tracker.update(self._get_detections())
        observation = self._build_obs()
        return observation

    def render(self, mode="human"):
        pass

    def enable_render(self):
        self._sim_wrap.enable_render()

    def _get_detections(self) -> Dict[int, np.ndarray]:
        detections = {k: np.array([v[0], v[1], 0., 0.])
                      for k, v in self._sim_wrap.sim_state.sensors["pedestrian_detector"].reading.pedestrians.items()}
        return detections

    @staticmethod
    def _build_robot_obs(robot_pose: np.ndarray, goal: np.ndarray) -> np.ndarray:
        return np.array([np.linalg.norm(goal[:2] - robot_pose[:2]),
                         goal[0] - robot_pose[0],
                         goal[1] - robot_pose[1],
                         robot_pose[2]]).astype(np.float32)

    def _build_peds_obs(self, robot_pose: np.ndarray,
                        current_poses: Dict[int, np.ndarray], predictions: Dict[int, np.ndarray]) -> \
            Tuple[np.ndarray, np.ndarray]:
        obs_ped_traj = np.ones((self._max_peds, self._ped_tracker.horizon + 1, 2)) * 100.
        obs_peds_ids = current_poses.keys()
        obs_peds_vis = np.zeros(self._max_peds, dtype=np.bool)
        for k in obs_peds_ids:
            obs_ped_traj[k, 0, :] = current_poses[k] - robot_pose[:2]
            obs_ped_traj[k, 1:, :] = predictions[k][0] - robot_pose[:2]
            obs_peds_vis[k] = True
        return obs_ped_traj, obs_peds_vis

    def _build_obs(self) -> Dict[str, np.ndarray]:
        goal = self._sim_wrap.goal
        robot_pose = self._sim_wrap.sim_state.world.robot.pose
        current_poses = self._ped_tracker.get_current_poses()
        predictions = {k: v[0] for k, v in self._ped_tracker.get_predictions().items()}

        robot_obs = SocialNavGraphEnv._build_robot_obs(robot_pose, goal)
        obs_ped_traj, obs_peds_vis = self._build_peds_obs(robot_obs, current_poses, predictions)

        return {
            "peds_traj": obs_ped_traj,
            "peds_visibility": obs_peds_vis,
            "robot_state": robot_obs
        }


@nip
class SocialNavGraphEnvFactory(AbstractEnvFactory):

    def __init__(self,
                 agents_sampler: AbstractAgentsSampler,
                 tracker_factory: Callable,
                 reward: AbstractReward,
                 sim_config: SimConfig,
                 render: bool = False,
                 normalize_actions: bool = False):
        self._agents_sampler = agents_sampler
        self._tracker_factory = tracker_factory
        self._reward = reward
        self._sim_config = sim_config
        self._render = render
        self._normalize_actions = normalize_actions

    def __call__(self) -> SocialNavGraphEnv:
        return SocialNavGraphEnv(self._agents_sampler,
                                 self._tracker_factory(),
                                 self._reward,
                                 self._sim_config,
                                 self._render,
                                 self._normalize_actions)
