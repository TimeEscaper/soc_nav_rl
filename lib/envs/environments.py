import random
import numpy as np
import gym

from abc import ABC, abstractmethod
from typing import Dict, Union, Optional, Tuple, List, Any, Callable, Type
from dataclasses import dataclass
from functools import partial
from nip import nip
from scipy.spatial.distance import cdist
from stable_baselines3.common.vec_env.base_vec_env import VecEnvIndices, VecEnvStepReturn

from lib.envs.rewards import AbstractReward, RewardContext
from lib.envs.agents_samplers import AbstractAgentsSampler, AgentsSample
from lib.envs.sim_config_samplers import AbstractActionSpaceConfig, AbstractProblemConfigSampler, SimConfig, \
    ProblemConfig
from lib.predictors.tracker import PedestrianTracker
from lib.utils.sampling import get_or_sample_uniform, get_or_sample_bool, get_or_sample_choice
from lib.envs.curriculum import AbstractCurriculum
from lib.controllers.controllers import AbstractController, AbstractControllerFactory
from lib.predictors.tracker import AbstractTrackerFactory

from pyminisim.core import PEDESTRIAN_RADIUS, ROBOT_RADIUS
from pyminisim.core import Simulation, SimulationState
from pyminisim.world_map import EmptyWorld, CirclesWorld
from pyminisim.robot import UnicycleRobotModel
from pyminisim.pedestrians import HeadedSocialForceModelPolicy, OptimalReciprocalCollisionAvoidance, \
    RandomWaypointTracker, FixedWaypointTracker
from pyminisim.sensors import PedestrianDetectorNoise, PedestrianDetector, PedestrianDetectorConfig, \
    LidarSensor, LidarSensorNoise
from pyminisim.visual import Renderer, CircleDrawing, AbstractDrawing, Covariance2dDrawing

from stable_baselines3.common.vec_env import VecEnv, SubprocVecEnv


class AbstractEnvFactory(ABC):

    @abstractmethod
    def __call__(self, n_envs: int, is_eval: bool) -> gym.Env:
        raise NotImplementedError()


class PyMiniSimBaseEnv:

    def __init__(self,
                 action_space_config: AbstractActionSpaceConfig,
                 sim_config: SimConfig,
                 peds_padding: int):
        assert action_space_config.action_type == AbstractActionSpaceConfig.TYPE_END2END, \
            f"End2end action supported in base env"

        self._action_space_config = action_space_config
        self._sim_config = sim_config
        self._render = sim_config.render
        self._peds_padding = peds_padding

        self._sim: Simulation = None
        self._renderer: Renderer = None
        self._robot_global_goal: np.ndarray = None
        self._goal_reach_threshold: float = None

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space_config.action_space

    @property
    def robot_global_goal(self) -> np.ndarray:
        return self._robot_global_goal

    @property
    def render_enabled(self) -> bool:
        return self._render

    @property
    def sim_state(self) -> SimulationState:
        return self._sim.current_state

    def step(self, action: np.ndarray) -> Tuple[bool, bool]:
        action = self._action_space_config.get_action(action)

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

        if has_collision:
            success = False
        else:
            success = np.linalg.norm(
                self._sim.current_state.world.robot.pose[:2] - self._robot_global_goal) \
                      - ROBOT_RADIUS < self._goal_reach_threshold

        return has_collision, success

    def reset(self, problem: ProblemConfig, agents: AgentsSample):
        self._goal_reach_threshold = problem.goal_reach_threshold
        self._robot_global_goal = agents.robot_goal

        robot_model = UnicycleRobotModel(initial_pose=agents.robot_initial_pose,
                                         initial_control=np.array([0.0, np.deg2rad(0.0)]))

        if problem.ped_model != "none" and agents.n_peds > 0:
            if agents.ped_goals is None:
                waypoint_tracker = RandomWaypointTracker(world_size=agents.world_size)
            else:
                waypoint_tracker = FixedWaypointTracker(initial_positions=agents.ped_initial_poses[:, :2],
                                                        waypoints=agents.ped_goals,
                                                        loop=True)

            if problem.ped_model == "hsfm":
                ped_model = HeadedSocialForceModelPolicy(waypoint_tracker=waypoint_tracker,
                                                         n_pedestrians=agents.n_peds,
                                                         initial_poses=agents.ped_initial_poses,
                                                         robot_visible=problem.robot_visible)
            elif problem.ped_model == "orca":
                ped_model = OptimalReciprocalCollisionAvoidance(dt=self._sim_config.sim_dt,
                                                                waypoint_tracker=waypoint_tracker,
                                                                n_pedestrians=agents.n_peds,
                                                                initial_poses=agents.ped_initial_poses,
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
            renderer.draw("goal", CircleDrawing(center=self._robot_global_goal[:2],
                                                radius=0.05,
                                                color=(255, 0, 0)))
        else:
            renderer = None

        self._sim = sim
        self._renderer = renderer

    def enable_render(self):
        self._render = True

    def build_observation(self) -> Dict[str, np.ndarray]:
        sensor_detections = self._sim.current_state.sensors["pedestrian_detector"].reading.pedestrians
        detections = np.ones((self._peds_padding, 2)) * 10000.
        ids = np.ones((self._peds_padding,)) * -1.

        robot_pose = self._sim.current_state.world.robot.pose
        robot_vel = self._sim.current_state.world.robot.velocity
        robot_obs = np.array([np.linalg.norm(self._robot_global_goal[:2] - robot_pose[:2]),
                              self._robot_global_goal[0] - robot_pose[0],
                              self._robot_global_goal[1] - robot_pose[1],
                              robot_pose[2],
                              robot_vel[0],
                              robot_vel[1],
                              robot_vel[2]])

        for i, (ped_id, ped_pose) in enumerate(sensor_detections.items()):
            detections[i] = np.array(ped_pose)
            ids[i] = ped_id
        return {
            "ped_detections": detections,
            "ped_ids": ids,
            "robot_obs": robot_obs,
            "robot_pose": robot_pose
        }

    def draw_predictions(self, means: np.ndarray, covs: np.ndarray, pred_mask: np.ndarray):
        if self._renderer is None:
            return
        if not pred_mask.any():
            self._renderer.clear_drawings(["pred_means", "pred_covs"])
            return
        means = means[np.where(pred_mask)]
        covs = covs[np.where(pred_mask)]
        self._renderer.draw("pred_means",
                            CircleDrawing(means.reshape((-1, 2)), 0.05, (173, 153, 121)))
        self._renderer.draw("pred_covs",
                            Covariance2dDrawing(means.reshape((-1, 2)), covs.reshape((-1, 2, 2)),
                                                (173, 153, 121), 0.05))


@nip
class End2EndSocialEnv(gym.Env):

    def __init__(self,
                 action_space_config: AbstractActionSpaceConfig,
                 sim_config: SimConfig,
                 curriculum: AbstractCurriculum,
                 reward: AbstractReward,
                 peds_padding: int,
                 prediction_horizon: int,
                 is_eval: bool):
        super(End2EndSocialEnv, self).__init__()

        self._base_env = PyMiniSimBaseEnv(action_space_config=action_space_config,
                                          sim_config=sim_config,
                                          peds_padding=peds_padding)
        self._curriculum = curriculum
        self._reward = reward
        self._peds_padding = peds_padding
        self._is_eval = is_eval

        self.observation_space = gym.spaces.Dict({
            "ped_detections": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(peds_padding, 2),
                dtype=np.float
            ),
            "ped_ids": gym.spaces.Box(
                low=-1,
                high=np.inf,
                shape=(peds_padding,),
                dtype=np.int
            ),
            "robot_obs": gym.spaces.Box(
                low=np.array([-np.inf, -np.inf, -np.inf, -np.pi, -np.inf, -np.inf, -np.inf]),
                high=np.array([np.inf, np.inf, np.inf, np.pi, np.inf, np.inf, np.inf]),
                shape=(7,),
                dtype=np.float
            ),
            "robot_pose": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(3,),
                dtype=np.float
            )
        })
        self.action_space = gym.spaces.Dict({
            "action": self._base_env.action_space,
            "pred_means": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(peds_padding, prediction_horizon, 2),
                dtype=np.float
            ),
            "pred_covs": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(peds_padding, prediction_horizon, 2, 2),
                dtype=np.float
            ),
            "pred_mask": gym.spaces.Box(
                low=False,
                high=True,
                shape=(peds_padding,),
                dtype=np.bool
            )
        })

        self._step_cnt = 0
        self._max_steps: int = None

    def step(self, action: Dict[str, np.ndarray]):
        pred_means = action["pred_means"]
        pred_covs = action["pred_covs"]
        pred_mask = action["pred_mask"]
        action = action["action"]

        reward_context = RewardContext()
        reward_context.set("goal", self._base_env.robot_global_goal)
        reward_context.set("previous_robot_pose", self._base_env.sim_state.world.robot.pose)
        reward_context.set("previous_ped_predictions", (pred_means, pred_covs, pred_mask))

        self._base_env.draw_predictions(pred_means, pred_covs, pred_mask)
        collision, success = self._base_env.step(action)
        self._step_cnt += 1
        truncated = (self._step_cnt >= self._max_steps) and not collision
        if truncated:
            success = False

        reward_context.set("robot_pose", self._base_env.sim_state.world.robot.pose)
        reward_context.set("robot_velocity", self._base_env.sim_state.world.robot.velocity)

        # https://stable-baselines3.readthedocs.io/en/master/common/logger.html#eval
        info = {"step_finished": True}
        if collision:
            done = True
            info.update({"done_reason": "collision",
                         "is_success": False})
            reward_context.set("collision", True)
        elif truncated:
            done = True
            info.update({"done_reason": "truncated",
                         "is_success": False,
                         "TimeLimit.truncated": True})  # https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html#tips-and-tricks-when-creating-a-custom-environment
            reward_context.set("truncated", True)
        elif success:
            done = True
            info.update({"done_reason": "success",
                         "is_success": True})
            reward_context.set("success", True)
        else:
            done = False

        reward, reward_info = self._reward(reward_context)
        info.update({"reward": reward_info})

        obs = self._base_env.build_observation()

        return obs, reward, done, info

    def reset(self) -> Dict[str, np.ndarray]:
        problem = self._curriculum.get_problem_sampler().sample() if not self._is_eval \
            else self._curriculum.get_eval_problem_sampler().sample()
        agents = self._curriculum.get_agents_sampler().sample() if not self._is_eval \
            else self._curriculum.get_eval_agents_sampler().sample()
        self._max_steps = problem.max_steps
        self._step_cnt = 0
        self._base_env.reset(problem, agents)
        return self._base_env.build_observation()

    def render(self, mode="human"):
        pass


@nip
class TrackerEnvWrapper(VecEnv):
    def __init__(self,
                 env: SubprocVecEnv,
                 tracker_factory: Callable[[], PedestrianTracker],
                 peds_padding: int,
                 rl_tracker_horizon: int):
        action_space = env.action_space["action"]
        observation_space = gym.spaces.Dict({
            "peds_traj": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(peds_padding, rl_tracker_horizon + 1, 2),  # Current state + predictions = 1 + horizon
                dtype=np.float
            ),
            "peds_visibility": gym.spaces.Box(
                low=False,
                high=True,
                shape=(peds_padding,),
                dtype=np.bool
            ),
            "robot_state": gym.spaces.Box(
                low=np.array([-np.inf, -np.inf, -np.inf, -np.pi, -np.inf, -np.inf, -np.inf]),
                high=np.array([np.inf, np.inf, np.inf, np.pi, np.inf, np.inf, np.inf]),
                shape=(7,),
                dtype=np.float
            )
        })
        VecEnv.__init__(self,
                        num_envs=env.num_envs,
                        observation_space=observation_space,
                        action_space=action_space)

        self._env = env
        self._n_envs = env.num_envs
        self._trackers = [tracker_factory() for _ in range(self._n_envs)]
        self._peds_padding = peds_padding
        self._rl_tracker_horizon = rl_tracker_horizon

        self.actions = None
        self._previous_obs = None

    def reset(self):
        base_obs = self._env.reset()
        self._update_trackers(base_obs, reset=sorted(range(len(self._trackers))))
        obs = self._build_obs(base_obs)
        self._previous_obs = obs
        return obs

    def render(self, mode="human"):
        pass

    def _get_prediction_action(self) -> List[Dict[str, np.ndarray]]:
        actions = []
        for tracker in self._trackers:
            pred_means = np.ones((self._peds_padding, tracker.horizon, 2)) * 1000.
            pred_covs = np.tile(np.eye(2), (self._peds_padding, tracker.horizon, 1, 1)) * 0.001
            pred_mask = np.zeros((self._peds_padding,), dtype=np.bool)
            for k, v in tracker.get_predictions().items():
                pred_means[k, :, :] = v[0]
                pred_covs[k, :, :, :] = v[1]
                pred_mask[k] = True
            actions.append({
                "pred_means": pred_means,
                "pred_covs": pred_covs,
                "pred_mask": pred_mask
            })
        return actions

    def _update_trackers(self, base_obs: Dict[str, np.ndarray], reset: Optional[List[int]] = None):
        for i, tracker in enumerate(self._trackers):
            if reset is not None and i in reset:
                tracker.reset()
            ped_detections = base_obs["ped_detections"][i]
            ped_ids = base_obs["ped_ids"][i]
            tracker_obs = {int(ped_ids[j]): np.array([ped_detections[j][0], ped_detections[j][1], 0, 0])
                           for j in range(ped_ids.shape[0]) if ped_ids[j] != -1}
            tracker.update(tracker_obs)

    def _build_obs(self, base_obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        new_obs = {
            "peds_traj": np.empty((self._n_envs, self._peds_padding, self._rl_tracker_horizon + 1, 2)),
            "peds_visibility": np.empty((self._n_envs, self._peds_padding)),
            "robot_state": np.empty((self._n_envs, 7))
        }
        for i, tracker in enumerate(self._trackers):
            robot_pose = base_obs["robot_pose"][i]
            obs_ped_traj, obs_ped_vis = self._get_prediction_obs(tracker, robot_pose)
            new_obs["peds_traj"][i] = obs_ped_traj
            new_obs["peds_visibility"][i] = obs_ped_vis
            new_obs["robot_state"][i] = base_obs["robot_obs"][i]
        return new_obs

    def _get_prediction_obs(self, tracker: PedestrianTracker, robot_pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        current_poses = tracker.get_current_poses()
        predictions = tracker.get_predictions()
        obs_ped_traj = np.ones((self._peds_padding, self._rl_tracker_horizon + 1, 2)) * 100.
        obs_peds_ids = current_poses.keys()
        obs_peds_vis = np.zeros(self._peds_padding, dtype=np.bool)
        for k in obs_peds_ids:
            obs_ped_traj[k, 0, :] = current_poses[k] - robot_pose[:2]
            obs_ped_traj[k, 1:, :] = predictions[k][0][:self._rl_tracker_horizon, :] - robot_pose[:2]
            obs_peds_vis[k] = True

        # TODO: Should we make soring optional?
        distances = np.linalg.norm(obs_ped_traj[:, 0, :], axis=1)
        sorted_indices = np.argsort(distances)
        obs_ped_traj = obs_ped_traj[sorted_indices]
        obs_peds_vis = obs_peds_vis[sorted_indices]

        return obs_ped_traj, obs_peds_vis

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        actions_list = self._get_prediction_action()
        for i, action_dict in enumerate(actions_list):
            action_dict["action"] = self.actions[i]

        base_obs, rewards, dones, infos = self._env.step(actions_list)

        trackers_to_reset = []
        for i in range(len(dones)):
            if dones[i]:
                trackers_to_reset.append(i)
                infos[i]["terminal_observation"] = {k: v[i] for k, v in self._previous_obs.items()}

        self._update_trackers(base_obs, reset=trackers_to_reset)
        obs = self._build_obs(base_obs)
        self._previous_obs = obs

        return obs, rewards, dones, infos

    def close(self) -> None:
        self._env.close()

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        return self._env.get_attr(attr_name, indices)

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        self._env.set_attr(attr_name, value, indices)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        return self._env.env_method(method_name, *method_args, indices, **method_kwargs)

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        return self._env.env_is_wrapped(wrapper_class, indices)

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        return self._env.seed(seed)


@nip
class End2EndPredictionEnvFactory(AbstractEnvFactory):

    def __init__(self,
                 action_space_config: AbstractActionSpaceConfig,
                 sim_config: SimConfig,
                 curriculum: AbstractCurriculum,
                 tracker_factory: AbstractTrackerFactory,
                 reward: AbstractReward,
                 peds_padding: int,
                 rl_tracker_horizon: int):
        self._action_space_config = action_space_config
        self._sim_config = sim_config
        self._curriculum = curriculum
        self._tracker_factory = tracker_factory
        self._reward = reward
        self._peds_padding = peds_padding
        self._rl_tracker_horizon = rl_tracker_horizon

    def __call__(self, n_envs: int, is_eval: bool) -> VecEnv:
        factory = partial(self._create_env, is_eval=is_eval)
        subproc_env = SubprocVecEnv([factory for _ in range(n_envs)])
        return TrackerEnvWrapper(env=subproc_env,
                                 tracker_factory=self._tracker_factory,
                                 peds_padding=self._peds_padding,
                                 rl_tracker_horizon=self._rl_tracker_horizon)

    def _create_env(self, is_eval: bool) -> gym.Env:
        return End2EndSocialEnv(action_space_config=self._action_space_config,
                                sim_config=self._sim_config,
                                curriculum=self._curriculum,
                                reward=self._reward,
                                peds_padding=self._peds_padding,
                                prediction_horizon=self._tracker_factory.horizon,
                                is_eval=is_eval)



# @nip
# class SocialNavGraphEnv(gym.Env):
#
#     def __init__(self,
#                  action_space_config: AbstractActionSpaceConfig,
#                  sim_config: SimConfig,
#                  curriculum: AbstractCurriculum,
#                  ped_tracker: PedestrianTracker,
#                  reward: AbstractReward,
#                  peds_padding: int,
#                  is_eval: bool,
#                  rl_tracker_horizon: int,
#                  controller: Optional[AbstractController] = None):
#         self._sim_wrap = PyMiniSimWrap(action_space_config,
#                                        sim_config,
#                                        curriculum,
#                                        ped_tracker,
#                                        is_eval,
#                                        controller)
#         self._reward = reward
#         self._rl_tracker_horizon = rl_tracker_horizon
#
#         self._peds_padding = peds_padding
#
#         self.observation_space = gym.spaces.Dict({
#             "peds_traj": gym.spaces.Box(
#                 low=-np.inf,
#                 high=np.inf,
#                 shape=(self._peds_padding, rl_tracker_horizon + 1, 2),  # Current state + predictions = 1 + horizon
#                 dtype=np.float
#             ),
#             "peds_visibility": gym.spaces.Box(
#                 low=False,
#                 high=True,
#                 shape=(self._peds_padding,),
#                 dtype=np.bool
#             ),
#             "robot_state": gym.spaces.Box(
#                 low=np.array([-np.inf, -np.inf, -np.inf, -np.pi, -np.inf, -np.inf, -np.inf]),
#                 high=np.array([np.inf, np.inf, np.inf, np.pi, np.inf, np.inf, np.inf]),
#                 shape=(7,),
#                 dtype=np.float
#             )
#         })
#
#         self.action_space = self._sim_wrap.action_space
#
#     def step(self, action: np.ndarray):
#         previous_robot_pose = self._sim_wrap.sim_state.world.robot.pose
#         previous_predictions = self._sim_wrap.ped_tracker.get_predictions()
#         goal = self._sim_wrap.goal
#
#         collision, truncated, success = self._sim_wrap.step(action)
#         robot_pose = self._sim_wrap.sim_state.world.robot.pose
#
#         reward_context = RewardContext()
#         reward_context.set("goal", goal)
#         reward_context.set("robot_pose", robot_pose)
#         reward_context.set("robot_velocity", self._sim_wrap.sim_state.world.robot.velocity)
#         reward_context.set("previous_robot_pose", previous_robot_pose)
#         reward_context.set("previous_ped_predictions", previous_predictions)
#
#         # https://stable-baselines3.readthedocs.io/en/master/common/logger.html#eval
#         if collision:
#             done = True
#             info = {"done_reason": "collision",
#                     "is_success": False}
#             reward_context.set("collision", True)
#         elif truncated:
#             done = True
#             info = {"done_reason": "truncated",
#                     "is_success": False,
#                     "TimeLimit.truncated": True}  # https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html#tips-and-tricks-when-creating-a-custom-environment
#             reward_context.set("truncated", True)
#         elif success:
#             done = True
#             info = {"done_reason": "success",
#                     "is_success": True}
#             reward_context.set("success", True)
#         else:
#             done = False
#             info = {}
#
#         reward, reward_info = self._reward(reward_context)
#         info.update({"reward": reward_info})
#
#         observation = self._build_obs()
#
#         return observation, reward, done, info
#
#     def reset(self):
#         self._sim_wrap.reset()
#         observation = self._build_obs()
#         return observation
#
#     def render(self, mode="human"):
#         pass
#
#     def enable_render(self):
#         self._sim_wrap.enable_render()
#
#     @staticmethod
#     def _build_robot_obs(robot_pose: np.ndarray, robot_vel: np.ndarray, goal: np.ndarray) -> np.ndarray:
#         return np.array([np.linalg.norm(goal[:2] - robot_pose[:2]),
#                          goal[0] - robot_pose[0],
#                          goal[1] - robot_pose[1],
#                          robot_pose[2],
#                          robot_vel[0],
#                          robot_vel[1],
#                          robot_vel[2]]).astype(np.float32)
#
#     def _build_peds_obs(self, robot_pose: np.ndarray,
#                         current_poses: Dict[int, np.ndarray], predictions: Dict[int, np.ndarray]) -> \
#             Tuple[np.ndarray, np.ndarray]:
#         obs_ped_traj = np.ones((self._peds_padding, self._rl_tracker_horizon + 1, 2)) * 100.
#         obs_peds_ids = current_poses.keys()
#         obs_peds_vis = np.zeros(self._peds_padding, dtype=np.bool)
#         for k in obs_peds_ids:
#             obs_ped_traj[k, 0, :] = current_poses[k] - robot_pose[:2]
#             obs_ped_traj[k, 1:, :] = predictions[k][:self._rl_tracker_horizon, :] - robot_pose[:2]
#             obs_peds_vis[k] = True
#
#         # TODO: Should we make soring optional?
#         distances = np.linalg.norm(obs_ped_traj[:, 0, :], axis=1)
#         sorted_indices = np.argsort(distances)
#         obs_ped_traj = obs_ped_traj[sorted_indices]
#         obs_peds_vis = obs_peds_vis[sorted_indices]
#
#         return obs_ped_traj, obs_peds_vis
#
#     def _build_obs(self) -> Dict[str, np.ndarray]:
#         goal = self._sim_wrap.goal
#         robot_pose = self._sim_wrap.sim_state.world.robot.pose
#         robot_vel = self._sim_wrap.sim_state.world.robot.velocity
#         current_poses = self._sim_wrap.ped_tracker.get_current_poses()
#         predictions = {k: v[0] for k, v in self._sim_wrap.ped_tracker.get_predictions().items()}
#
#         robot_obs = SocialNavGraphEnv._build_robot_obs(robot_pose, robot_vel, goal)
#         obs_ped_traj, obs_peds_vis = self._build_peds_obs(robot_pose, current_poses, predictions)
#
#         return {
#             "peds_traj": obs_ped_traj,
#             "peds_visibility": obs_peds_vis,
#             "robot_state": robot_obs
#         }
#
#
# @nip
# class SocialNavGraphEnvFactory(AbstractEnvFactory):
#
#     def __init__(self,
#                  action_space_config: AbstractActionSpaceConfig,
#                  sim_config: SimConfig,
#                  curriculum: AbstractCurriculum,
#                  tracker_factory: Callable,
#                  reward: AbstractReward,
#                  peds_padding: int,
#                  rl_tracker_horizon: int,
#                  controller_factory: Optional[AbstractControllerFactory] = None):
#         self._action_space_config = action_space_config
#         self._sim_config = sim_config
#         self._curriculum = curriculum
#         self._ped_tracker_factory = tracker_factory
#         self._reward = reward
#         self._peds_padding = peds_padding
#         self._rl_tracking_horizon = rl_tracker_horizon
#         self._controller_factory = controller_factory
#
#     def __call__(self, is_eval: bool) -> SocialNavGraphEnv:
#         controller = self._controller_factory() if self._controller_factory is not None else None
#         return SocialNavGraphEnv(action_space_config=self._action_space_config,
#                                  sim_config=self._sim_config,
#                                  curriculum=self._curriculum,
#                                  ped_tracker=self._ped_tracker_factory(),
#                                  reward=self._reward,
#                                  peds_padding=self._peds_padding,
#                                  rl_tracker_horizon=self._rl_tracking_horizon,
#                                  controller=controller,
#                                  is_eval=is_eval)
