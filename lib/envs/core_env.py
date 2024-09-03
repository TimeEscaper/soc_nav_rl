from typing import Union, Tuple, List

import numpy as np
from pyminisim.core import PEDESTRIAN_RADIUS, ROBOT_RADIUS
from pyminisim.core import Simulation, SimulationState
from pyminisim.pedestrians import HeadedSocialForceModelPolicy, ORCAParams, ORCAPedestriansModel, \
    RandomWaypointTracker, FixedWaypointTracker, ReplayPedestriansPolicy
from pyminisim.robot import UnicycleRobotModel
from pyminisim.sensors import PedestrianDetector, PedestrianDetectorConfig
from pyminisim.visual import Renderer, CircleDrawing, AbstractDrawing
from pyminisim.world_map import EmptyWorld

from lib.envs.curriculum import AbstractCurriculum
from lib.envs.sim_config_samplers import SimConfig, ProblemConfig
from lib.envs.agents_samplers import AgentsSample


class PyMiniSimCoreEnv:

    def __init__(self,
                 sim_config: SimConfig,
                 curriculum: AbstractCurriculum,
                 is_eval: bool):

        self._sim_config = sim_config
        self._curriculum = curriculum
        self._render = sim_config.render
        self._is_eval = is_eval

        self._sim: Simulation = None
        self._renderer: Renderer = None
        self._robot_goal: np.ndarray = None
        self._goal_reach_threshold: float = None
        self._problem_config: ProblemConfig = None

        self._id = np.random.randint(1, 150)

    def update_curriculum(self):
        self._curriculum.update_stage()

    @property
    def goal(self) -> np.ndarray:
        return self._robot_goal.copy()

    @property
    def render_enabled(self) -> bool:
        return self._render

    @property
    def sim_state(self) -> SimulationState:
        return self._sim.current_state

    @property
    def problem_config(self) -> ProblemConfig:
        return self._problem_config

    def step(self, action: np.ndarray) -> Tuple[bool, bool, float]:
        hold_time = 0.
        has_collision = False
        min_separation_distance = np.inf

        while hold_time < self._sim_config.policy_dt:
            self._sim.step(action)
            hold_time += self._sim_config.sim_dt
            if self._renderer is not None:
                self._renderer.render()
            collisions = self._sim.current_state.world.robot_to_pedestrians_collisions
            has_collision = collisions is not None and len(collisions) > 0

            robot_position = self._sim.current_state.world.robot.pose[:2]
            if self._sim.current_state.world.pedestrians is not None:
                ped_positions = np.array([v[:2] for v in self._sim.current_state.world.pedestrians.poses.values()])
                if ped_positions.shape[0] > 0:
                    separation = np.min(
                        np.linalg.norm(robot_position - ped_positions, axis=1)) - ROBOT_RADIUS - PEDESTRIAN_RADIUS
                    if separation < min_separation_distance:
                        min_separation_distance = separation

            if has_collision:
                break

        goal_reached = np.linalg.norm(self._sim.current_state.world.robot.pose[:2] -
                                      self._robot_goal) - ROBOT_RADIUS < self._goal_reach_threshold

        return has_collision, goal_reached, min_separation_distance

    def reset(self):
        # print(f"Resetting, stage: {self._curriculum.get_current_stage()[0]}, "
        #       f"is_eval: {self._is_eval}, id: {self._id}")

        problem = self._curriculum.get_problem_sampler().sample() if not self._is_eval \
            else self._curriculum.get_eval_problem_sampler().sample()
        self._problem_config = problem
        self._goal_reach_threshold = problem.goal_reach_threshold

        agents_sample = self._curriculum.get_agents_sampler().sample() if not self._is_eval \
            else self._curriculum.get_eval_agents_sampler().sample()
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
                                                         robot_visible=problem.robot_visible,
                                                         pedestrian_linear_velocity_magnitude=agents_sample.ped_linear_vels)
            elif problem.ped_model == "orca":
                # TODO: Implement velocities in ORCA
                ped_model = ORCAPedestriansModel(dt=self._sim_config.sim_dt,
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

    def enable_render(self):
        self._render = True

    def draw(self, name: str, drawing: AbstractDrawing):
        if self._renderer is not None:
            self._renderer.draw(name, drawing)

    def clear_drawing(self, name: Union[str, List[str]]):
        if self._renderer is not None:
            self._renderer.clear_drawings(name)


class PyMiniSimCoreReplayEnv:

    def __init__(self,
                 sim_config: SimConfig,
                 curriculum: AbstractCurriculum,
                 is_eval: bool,
                 max_steps: int):

        self._sim_config = sim_config
        self._curriculum = curriculum
        self._render = sim_config.render
        self._is_eval = is_eval
        self._max_steps = max_steps

        self._sim: Simulation = None
        self._renderer: Renderer = None
        self._robot_goal: np.ndarray = None
        self._goal_reach_threshold: float = None
        self._problem_config: ProblemConfig = None

        self._id = np.random.randint(1, 150)

    def update_curriculum(self):
        self._curriculum.update_stage()

    @property
    def goal(self) -> np.ndarray:
        return self._robot_goal.copy()

    @property
    def render_enabled(self) -> bool:
        return self._render

    @property
    def sim_config(self) -> SimConfig:
        return self._sim_config

    @property
    def sim_state(self) -> SimulationState:
        return self._sim.current_state

    @property
    def problem_config(self) -> ProblemConfig:
        return self._problem_config

    def step(self, action: np.ndarray) -> Tuple[bool, bool, float]:
        hold_time = 0.
        has_collision = False
        min_separation_distance = np.inf

        while hold_time < self._sim_config.policy_dt:
            self._sim.step(action)
            hold_time += self._sim_config.sim_dt
            if self._renderer is not None:
                self._renderer.render()
            collisions = self._sim.current_state.world.robot_to_pedestrians_collisions
            has_collision = collisions is not None and len(collisions) > 0

            robot_position = self._sim.current_state.world.robot.pose[:2]
            if self._sim.current_state.world.pedestrians is not None:
                ped_positions = np.array([v[:2] for v in self._sim.current_state.world.pedestrians.poses.values()])
                if ped_positions.shape[0] > 0:
                    separation = np.min(
                        np.linalg.norm(robot_position - ped_positions, axis=1)) - ROBOT_RADIUS - PEDESTRIAN_RADIUS
                    if separation < min_separation_distance:
                        min_separation_distance = separation

            if has_collision:
                break

        goal_reached = np.linalg.norm(self._sim.current_state.world.robot.pose[:2] -
                                      self._robot_goal) - ROBOT_RADIUS < self._goal_reach_threshold

        return has_collision, goal_reached, min_separation_distance

    def reset(self):
        # print(f"Resetting, stage: {self._curriculum.get_current_stage()[0]}, "
        #       f"is_eval: {self._is_eval}, id: {self._id}")

        problem = self._curriculum.get_problem_sampler().sample() if not self._is_eval \
            else self._curriculum.get_eval_problem_sampler().sample()
        self._problem_config = problem
        self._goal_reach_threshold = problem.goal_reach_threshold

        agents_sample = self._curriculum.get_agents_sampler().sample() if not self._is_eval \
            else self._curriculum.get_eval_agents_sampler().sample()
        self._robot_goal = agents_sample.robot_goal

        sim, renderer = self._create_sim(problem, agents_sample)
        if problem.ped_model != "none" and agents_sample.n_peds > 0:
            replay_trajectories = self._collect_trajectories(sim)
            sim, renderer = self._create_sim(problem, agents_sample, replay_trajectories)

        self._sim = sim
        self._renderer = renderer

    def enable_render(self):
        self._render = True

    def get_future_poses(self, horizon: int) -> np.ndarray | None:
        ped_model = self._sim._pedestrians_model
        if not isinstance(ped_model, ReplayPedestriansPolicy):
            return None
        current_step = ped_model._current_step
        return ped_model._poses[current_step:current_step + horizon + 1, :, :]

    def draw(self, name: str, drawing: AbstractDrawing):
        if self._renderer is not None:
            self._renderer.draw(name, drawing)

    def clear_drawing(self, name: Union[str, List[str]]):
        if self._renderer is not None:
            self._renderer.clear_drawings(name)

    def _create_sim(self,
                    problem: ProblemConfig,
                    agents_sample: AgentsSample,
                    replay_trajectories: tuple[np.ndarray, np.ndarray] | None = None) -> tuple[Simulation, Renderer]:
        
        if replay_trajectories is None:
            robot_model = None
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
                                                            robot_visible=problem.robot_visible,
                                                            pedestrian_linear_velocity_magnitude=agents_sample.ped_linear_vels)
                elif problem.ped_model == "orca":
                    # TODO: Implement velocities in ORCA
                    ped_model = ORCAPedestriansModel(dt=self._sim_config.sim_dt,
                                                                    waypoint_tracker=waypoint_tracker,
                                                                    n_pedestrians=agents_sample.n_peds,
                                                                    initial_poses=agents_sample.ped_initial_poses,
                                                                    params=ORCAParams(default_max_speed=np.max(agents_sample.ped_linear_vels) + 0.1),
                                                                    preferred_speeds=agents_sample.ped_linear_vels,
                                                                    robot_visible=problem.robot_visible)
                else:
                    raise ValueError()
            else:
                ped_model = None
        else:
            robot_model = UnicycleRobotModel(initial_pose=agents_sample.robot_initial_pose,
                                             initial_control=np.array([0., 0.]))
            ped_model = ReplayPedestriansPolicy(replay_trajectories[0], replay_trajectories[1])

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

        if self._render and replay_trajectories is not None:
            renderer = Renderer(simulation=sim,
                                resolution=70.,
                                screen_size=(800, 800))
            renderer.draw("goal", CircleDrawing(center=self._robot_goal[:2],
                                                radius=0.05,
                                                color=(255, 0, 0)))
        else:
            renderer = None

        return sim, renderer
    
    def _collect_trajectories(self, sim: Simulation) -> tuple[np.ndarray, np.ndarray]:
        poses = []
        velocities = []
        max_steps = (int((self._sim_config.policy_dt / sim.sim_dt)) + 1) * self._max_steps
        step_count = 0

        while step_count < max_steps:
            current_poses = sim.current_state.world.pedestrians.poses
            current_velocities = sim.current_state.world.pedestrians.velocities
            poses.append([current_poses[k] for k in sorted(current_poses.keys())])
            velocities.append([current_velocities[k] for k in sorted(current_velocities.keys())])
            sim.step(None)
            step_count += 1

        poses = np.array(poses)
        velocities = np.array(velocities)

        return poses, velocities

