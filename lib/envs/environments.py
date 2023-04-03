import random
import numpy as np
import gym

from abc import ABC, abstractmethod
from typing import Dict, Union, Optional, Tuple, List, Any, Callable
from dataclasses import dataclass
from nip import nip
from scipy.spatial.distance import cdist
from lib.envs.rewards import AbstractReward, RewardContext
from lib.envs.agents_samplers import AbstractAgentsSampler
from lib.envs.sim_config_samplers import AbstractActionSpaceConfig, AbstractProblemConfigSampler, SimConfig, \
    ProblemConfig
from lib.predictors.tracker import PedestrianTracker
from lib.utils.sampling import get_or_sample_uniform, get_or_sample_bool, get_or_sample_choice
from lib.envs.curriculum import AbstractCurriculum

from pyminisim.core import PEDESTRIAN_RADIUS, ROBOT_RADIUS
from pyminisim.core import Simulation, SimulationState
from pyminisim.world_map import EmptyWorld, CirclesWorld
from pyminisim.robot import UnicycleRobotModel
from pyminisim.pedestrians import HeadedSocialForceModelPolicy, OptimalReciprocalCollisionAvoidance, \
    RandomWaypointTracker, FixedWaypointTracker
from pyminisim.sensors import PedestrianDetectorNoise, PedestrianDetector, PedestrianDetectorConfig, \
    LidarSensor, LidarSensorNoise
from pyminisim.visual import Renderer, CircleDrawing, AbstractDrawing


class AbstractEnvFactory(ABC):

    @abstractmethod
    def __call__(self) -> gym.Env:
        raise NotImplementedError()


class PyMiniSimWrap:

    def __init__(self,
                 action_space_config: AbstractActionSpaceConfig,
                 sim_config: SimConfig,
                 curriculum: AbstractCurriculum):
        self._action_space_config = action_space_config
        self._sim_config = sim_config
        self._curriculum = curriculum
        self._render = sim_config.render

        self._step_cnt = 0

        self._sim: Simulation = None
        self._renderer: Renderer = None
        self._robot_goal: np.ndarray = None
        self._goal_reach_threshold: float = None
        self._max_steps: int = None

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space_config.action_space

    @property
    def goal(self) -> np.ndarray:
        return self._robot_goal

    @property
    def current_step_cnt(self) -> int:
        return self._step_cnt

    @property
    def render_enabled(self) -> bool:
        return self._render

    @property
    def sim_state(self) -> SimulationState:
        return self._sim.current_state

    def step(self, action: np.ndarray) -> Tuple[bool, bool, bool]:
        action = self._action_space_config.get_control(action)

        hold_time = 0.
        has_collision = False
        while hold_time < self._sim_config.policy_dt:
            self._sim.step(action)
            hold_time += self._sim_config.sim_dt
            if self._renderer is not None:
                self._renderer.render()
            collisions = self._sim.current_state.world.robot_to_pedestrians_collisions
            has_collision = collisions is not None and len(collisions) > 0
            if has_collision:
                break

        self._step_cnt += 1
        truncated = (self._step_cnt >= self._max_steps) and not has_collision

        if has_collision or truncated:
            success = False
        else:
            success = np.linalg.norm(
                self._sim.current_state.world.robot.pose[:2] - self._robot_goal) \
                      - ROBOT_RADIUS < self._goal_reach_threshold

        return has_collision, truncated, success

    def reset(self):
        problem = self._curriculum.get_problem_sampler().sample()
        self._goal_reach_threshold = problem.goal_reach_threshold
        self._max_steps = problem.max_steps

        agents_sample = self._curriculum.get_agents_sampler().sample()
        self._robot_goal = agents_sample.robot_goal

        robot_model = UnicycleRobotModel(initial_pose=agents_sample.robot_initial_pose,
                                         initial_control=np.array([0.0, np.deg2rad(0.0)]))

        if problem.ped_model != "none" and agents_sample.n_peds > 0:
            if agents_sample.ped_goals is None:
                waypoint_tracker = RandomWaypointTracker(world_size=agents_sample.world_size)
            else:
                waypoint_tracker = FixedWaypointTracker(initial_positions=agents_sample.ped_initial_poses[:, :2],
                                                        waypoints=agents_sample.ped_goals,
                                                        loop=True)

            if problem.ped_model == "hsfm":
                ped_model = HeadedSocialForceModelPolicy(waypoint_tracker=waypoint_tracker,
                                                         n_pedestrians=agents_sample.n_peds,
                                                         initial_poses=agents_sample.ped_initial_poses,
                                                         robot_visible=problem.robot_visible)
            elif problem.ped_model == "orca":
                ped_model = OptimalReciprocalCollisionAvoidance(dt=self._sim_config.sim_dt,
                                                                waypoint_tracker=waypoint_tracker,
                                                                n_pedestrians=agents_sample.n_peds,
                                                                initial_poses=agents_sample.ped_initial_poses,
                                                                robot_visible=problem.robot_visible)
            else:
                raise ValueError()
        else:
            ped_model = None

        ped_detector = PedestrianDetector(
            config=PedestrianDetectorConfig(max_dist=problem.detector_range,
                                            fov=problem.detector_fov,
                                            return_type=PedestrianDetectorConfig.RETURN_ABSOLUTE))

        sim = Simulation(world_map=EmptyWorld(),
                         robot_model=robot_model,
                         pedestrians_model=ped_model,
                         sensors=[ped_detector],
                         sim_dt=self._sim_config.sim_dt,
                         rt_factor=self._sim_config.rt_factor)
        if self._render:
            renderer = Renderer(simulation=sim,
                                resolution=70.,
                                screen_size=(800, 800))
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
class SocialNavGraphEnv(gym.Env):

    def __init__(self,
                 action_space_config: AbstractActionSpaceConfig,
                 sim_config: SimConfig,
                 curriculum: AbstractCurriculum,
                 ped_tracker: PedestrianTracker,
                 reward: AbstractReward,
                 peds_padding: int):
        self._sim_wrap = PyMiniSimWrap(action_space_config,
                                       sim_config,
                                       curriculum)
        self._reward = reward
        self._ped_tracker = ped_tracker

        self._peds_padding = peds_padding

        self._previous_ped_predictions = ped_tracker.get_predictions()

        self.observation_space = gym.spaces.Dict({
            "peds_traj": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self._peds_padding, ped_tracker.horizon + 1, 2),  # Current state + predictions = 1 + horizon
                dtype=np.float
            ),
            "peds_visibility": gym.spaces.Box(
                low=False,
                high=True,
                shape=(self._peds_padding,),
                dtype=np.bool
            ),
            "robot_state": gym.spaces.Box(
                low=np.array([-np.inf, -np.inf, -np.inf, -np.pi, -np.inf, -np.inf, -np.inf]),
                high=np.array([np.inf, np.inf, np.inf, np.pi, np.inf, np.inf, np.inf]),
                shape=(7,),
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

        # https://stable-baselines3.readthedocs.io/en/master/common/logger.html#eval
        if collision:
            done = True
            info = {"done_reason": "collision",
                    "is_success": False}
            reward_context.set("collision", True)
        elif truncated:
            done = True
            info = {"done_reason": "truncated",
                    "is_success": False,
                    "TimeLimit.truncated": True}  # https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html#tips-and-tricks-when-creating-a-custom-environment
            reward_context.set("truncated", True)
        elif success:
            done = True
            info = {"done_reason": "success",
                    "is_success": True}
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
    def _build_robot_obs(robot_pose: np.ndarray, robot_vel: np.ndarray, goal: np.ndarray) -> np.ndarray:
        return np.array([np.linalg.norm(goal[:2] - robot_pose[:2]),
                         goal[0] - robot_pose[0],
                         goal[1] - robot_pose[1],
                         robot_pose[2],
                         robot_vel[0],
                         robot_vel[1],
                         robot_vel[2]]).astype(np.float32)

    def _build_peds_obs(self, robot_pose: np.ndarray,
                        current_poses: Dict[int, np.ndarray], predictions: Dict[int, np.ndarray]) -> \
            Tuple[np.ndarray, np.ndarray]:
        obs_ped_traj = np.ones((self._peds_padding, self._ped_tracker.horizon + 1, 2)) * 100.
        obs_peds_ids = current_poses.keys()
        obs_peds_vis = np.zeros(self._peds_padding, dtype=np.bool)
        for k in obs_peds_ids:
            obs_ped_traj[k, 0, :] = current_poses[k] - robot_pose[:2]
            obs_ped_traj[k, 1:, :] = predictions[k][0] - robot_pose[:2]
            obs_peds_vis[k] = True

        # TODO: Should we make soring optional?
        distances = np.linalg.norm(obs_ped_traj[:, 0, :], axis=1)
        sorted_indices = np.argsort(distances)
        obs_ped_traj = obs_ped_traj[sorted_indices]
        obs_peds_vis = obs_peds_vis[sorted_indices]

        return obs_ped_traj, obs_peds_vis

    def _build_obs(self) -> Dict[str, np.ndarray]:
        goal = self._sim_wrap.goal
        robot_pose = self._sim_wrap.sim_state.world.robot.pose
        robot_vel = self._sim_wrap.sim_state.world.robot.velocity
        current_poses = self._ped_tracker.get_current_poses()
        predictions = {k: v[0] for k, v in self._ped_tracker.get_predictions().items()}

        robot_obs = SocialNavGraphEnv._build_robot_obs(robot_pose, robot_vel, goal)
        obs_ped_traj, obs_peds_vis = self._build_peds_obs(robot_pose, current_poses, predictions)

        return {
            "peds_traj": obs_ped_traj,
            "peds_visibility": obs_peds_vis,
            "robot_state": robot_obs
        }


@nip
class SocialNavGraphEnvFactory(AbstractEnvFactory):

    def __init__(self,
                 action_space_config: AbstractActionSpaceConfig,
                 sim_config: SimConfig,
                 curriculum: AbstractCurriculum,
                 tracker_factory: Callable,
                 reward: AbstractReward,
                 peds_padding: int):
        self._action_space_config = action_space_config
        self._sim_config = sim_config
        self._curriculum = curriculum
        self._ped_tracker_factory = tracker_factory
        self._reward = reward
        self._peds_padding = peds_padding

    def __call__(self) -> SocialNavGraphEnv:
        return SocialNavGraphEnv(action_space_config=self._action_space_config,
                                 sim_config=self._sim_config,
                                 curriculum=self._curriculum,
                                 ped_tracker=self._ped_tracker_factory(),
                                 reward=self._reward,
                                 peds_padding=self._peds_padding)
