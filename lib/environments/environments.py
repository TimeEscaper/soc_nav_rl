import random
import numpy as np
import gym

from abc import ABC, abstractmethod
from typing import Dict, Union, Optional, Tuple, List, Any
from dataclasses import dataclass
from nip import nip
from scipy.spatial.distance import cdist
from lib.environments.rewards import AbstractReward

from pyminisim.core import PEDESTRIAN_RADIUS, ROBOT_RADIUS
from pyminisim.core import Simulation
from pyminisim.world_map import EmptyWorld, CirclesWorld
from pyminisim.robot import UnicycleRobotModel
from pyminisim.pedestrians import HeadedSocialForceModelPolicy, OptimalReciprocalCollisionAvoidance, \
    RandomWaypointTracker, FixedWaypointTracker
from pyminisim.sensors import PedestrianDetectorNoise, PedestrianDetector, PedestrianDetectorConfig, \
    LidarSensor, LidarSensorNoise
from pyminisim.visual import Renderer, CircleDrawing


@dataclass
class AgentsSample:
    n_peds: int
    robot_initial_pose: np.ndarray
    robot_goal: np.ndarray
    ped_initial_poses: np.ndarray
    world_size: Tuple[int, int]
    ped_goals: Optional[np.ndarray] = None


class AbstractAgentsSampler(ABC):

    @abstractmethod
    def sample(self) -> AgentsSample:
        raise NotImplementedError()


@nip
class CompositeAgentsSampler(AbstractAgentsSampler):

    def __init__(self, samplers: List[AbstractAgentsSampler]):
        super(CompositeAgentsSampler, self).__init__()
        assert len(samplers) > 0
        self._samplers = tuple(samplers)

    def sample(self) -> AgentsSample:
        sampler = random.choice(self._samplers)
        return sampler.sample()


@nip
class RandomPedSampler(AbstractAgentsSampler):

    def __init__(self, n_peds: Union[int, Tuple[int, int]], sampling_square: Tuple[int, int] = (20, 20),
                 min_robot_goal_distance: float = 5., max_sample_trials: int = 1000):
        super(RandomPedSampler, self).__init__()
        assert len(n_peds) == 2 and n_peds[1] > n_peds[0] > 0
        self._n_peds = n_peds
        self._sampling_square = sampling_square
        self._min_robot_goal_distance = min_robot_goal_distance
        self._max_sample_trials = max_sample_trials

    def sample(self) -> AgentsSample:
        if not isinstance(self._n_peds, int):
            n_peds = np.random.randint(self._n_peds[0], self._n_peds[1])
        else:
            n_peds = self._n_peds

        low = np.array([-self._sampling_square[0] / 2, -self._sampling_square[1] / 2])
        high = np.array([self._sampling_square[0] / 2, self._sampling_square[1] / 2])
        sampled = False
        for _ in range(self._max_sample_trials):
            positions = np.random.uniform(low, high, (n_peds + 1, 2))
            dists = cdist(positions, positions, "euclidean")
            dists = dists[np.triu_indices(dists.shape[0], 1)]
            sampled = (dists > PEDESTRIAN_RADIUS + ROBOT_RADIUS).all()
        if not sampled:
            raise RuntimeError("Failed to sample positions")
        ped_poses = positions[:-1, :]
        robot_pose = positions[-1, :]

        sampled = False
        for _ in range(self._max_sample_trials):
            goal = np.random.uniform(low, high)
            sampled = np.linalg.norm(robot_pose - goal) >= self._min_robot_goal_distance
        if not sampled:
            raise RuntimeError("Failed to sample goal")

        ped_poses = np.concatenate((ped_poses, np.random.uniform(-np.pi, np.pi, (n_peds, 1))), axis=-1)
        robot_pose = np.concatenate((robot_pose, np.random.uniform(-np.pi, np.pi, (1,))), axis=-1)

        return AgentsSample(n_peds=self._n_peds,
                            robot_initial_pose=robot_pose,
                            robot_goal=goal,
                            world_size=self._sampling_square,
                            ped_initial_poses=ped_poses,
                            robot_visible=self._robot_visible)


class PyMiniSimEnvBase(gym.Env, ABC):

    def __init__(self,
                 ped_model: str,
                 agents_sampler: AbstractAgentsSampler,
                 reward: AbstractReward,
                 sim_dt: float = 0.01,
                 policy_dt: float = 0.1,
                 robot_visible: bool = False,
                 render: bool = False,
                 detector_range: float = 5.,
                 detector_fov: float = 140.,
                 control_lb: np.ndarray = np.array([0., 0.]),
                 control_ub: np.ndarray = np.array([2., 0.9 * np.pi]),
                 goal_reach_threshold: float = 0.1,
                 max_steps: int = 300):
        assert ped_model in ("hsfm", "orca")
        self._ped_model = ped_model
        self._agents_sampler = agents_sampler
        self._reward = reward
        self._sim_dt = sim_dt
        self._policy_dt = policy_dt
        self._robot_visible = robot_visible
        self._render = render
        self._detector_range = detector_range
        self._detector_fov = detector_fov
        self._control_lb = control_lb.copy()
        self._control_ub = control_ub.copy()
        self._goal_reach_threshold = goal_reach_threshold
        self._max_steps = max_steps

        self._step_cnt = 0

        self._sim: Simulation = None
        self._renderer: Renderer = None
        self._robot_goal: np.ndarray = None

    def step(self, action: np.ndarray):
        action = np.clip(action, self._control_lb, self._control_ub)
        sim = self._sim
        assert sim is not None, "Reset method must be called before first call of the step method"

        hold_time = 0.
        has_collision = False
        while hold_time < self._policy_dt:
            sim.step(action)
            hold_time += self._sim_dt
            if self._renderer is not None:
                self._renderer.render()
            collisions = sim.current_state.world.robot_to_pedestrians_collisions
            has_collision = collisions is not None and len(collisions) > 0
            if has_collision:
                break
        self._step_cnt += 1

        if has_collision:
            done = True
            info = {"done_reason": "collision"}
        elif self._step_cnt > self._max_steps:
            done = True
            info = {"done_reason": "truncated"}
        else:
            done = np.linalg.norm(
                sim.current_state.world.robot.pose[:2] - self._robot_goal) - ROBOT_RADIUS < self._goal_reach_threshold
            info = {}
            if done:
                info["done_reason"] = "success"

    def reset(self):
        agents_sample = self._agents_sampler.sample()
        self._robot_goal = agents_sample.robot_goal

        robot_model = UnicycleRobotModel(initial_pose=agents_sample.robot_initial_pose,
                                         initial_control=np.array([0.0, np.deg2rad(25.0)]))

        if agents_sample.ped_goals is None:
            waypoint_tracker = RandomWaypointTracker(world_size=agents_sample.world_size)
        else:
            waypoint_tracker = FixedWaypointTracker(waypoints=agents_sample.ped_goals[np.newaxis, :, :])

        if self._ped_model == "hsfm":
            ped_model = HeadedSocialForceModelPolicy(waypoint_tracker=waypoint_tracker,
                                                     n_pedestrians=agents_sample.n_peds,
                                                     initial_poses=agents_sample.ped_initial_poses,
                                                     robot_visible=self._robot_visible)
        elif self._ped_model == "orca":
            ped_model = OptimalReciprocalCollisionAvoidance(dt=self._sim_dt,
                                                            waypoint_tracker=waypoint_tracker,
                                                            n_pedestrians=agents_sample.n_peds,
                                                            initial_poses=agents_sample.ped_initial_poses,
                                                            robot_visible=self._robot_visible)
        else:
            raise ValueError()

        ped_detector = PedestrianDetector(config=PedestrianDetectorConfig(max_dist=self._detector_range,
                                                                          fov=self._detector_fov))

        sim = Simulation(world_map=EmptyWorld(),
                         robot_model=robot_model,
                         pedestrians_model=ped_model,
                         sensors=[ped_detector],
                         sim_dt=self._sim_dt)
        if self._render:
            renderer = Renderer(simulation=sim,
                                resolution=30.,
                                screen_size=(1500, 1500))
        else:
            renderer = None

        self._sim = sim
        self._renderer = renderer

        self._step_cnt = 0

    def render(self, mode="human"):
        pass

    def _build_observation(self) -> Any:
        raise NotImplementedError()

    def _reset_internal(self):
        raise NotImplementedError()
