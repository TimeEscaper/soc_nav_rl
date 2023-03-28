import random
import numpy as np

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Tuple, Optional, List, Union
from nip import nip
from scipy.spatial.distance import cdist
from pyminisim.core import ROBOT_RADIUS, PEDESTRIAN_RADIUS


@dataclass
class AgentsSample:
    n_peds: int
    robot_initial_pose: np.ndarray
    robot_goal: np.ndarray
    ped_initial_poses: np.ndarray
    world_size: Tuple[int, int]
    ped_goals: Optional[np.ndarray] = None


class AbstractAgentsSampler(ABC):

    def __init__(self, max_peds: int):
        self._max_peds = max_peds

    @property
    def max_peds(self) -> int:
        return self._max_peds

    @abstractmethod
    def sample(self) -> AgentsSample:
        raise NotImplementedError()


@nip
class CompositeAgentsSampler(AbstractAgentsSampler):

    def __init__(self, samplers: List[AbstractAgentsSampler]):
        super(CompositeAgentsSampler, self).__init__(
            max_peds=max([sampler.max_peds for sampler in samplers])
        )
        assert len(samplers) > 0
        self._samplers = tuple(samplers)

    def sample(self) -> AgentsSample:
        sampler = random.choice(self._samplers)
        return sampler.sample()


@nip
class RandomAgentsSampler(AbstractAgentsSampler):

    def __init__(self, n_peds: Union[int, Tuple[int, int]], sampling_square: Tuple[int, int] = (20, 20),
                 min_robot_goal_distance: float = 5., max_sample_trials: int = 1000):
        super(RandomAgentsSampler, self).__init__(
            max_peds=n_peds[1] if isinstance(n_peds, tuple) else n_peds
        )
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
            if sampled:
                break
        if not sampled:
            raise RuntimeError("Failed to sample positions")
        ped_poses = positions[:-1, :]
        robot_pose = positions[-1, :]

        sampled = False
        for _ in range(self._max_sample_trials):
            goal = np.random.uniform(low, high)
            sampled = np.linalg.norm(robot_pose - goal) >= self._min_robot_goal_distance
            if sampled:
                break
        if not sampled:
            raise RuntimeError("Failed to sample goal")

        ped_poses = np.concatenate((ped_poses, np.random.uniform(-np.pi, np.pi, (n_peds, 1))), axis=-1)
        robot_pose = np.concatenate((robot_pose, np.random.uniform(-np.pi, np.pi, (1,))), axis=-1)

        return AgentsSample(n_peds=n_peds,
                            robot_initial_pose=robot_pose,
                            robot_goal=goal,
                            world_size=self._sampling_square,
                            ped_initial_poses=ped_poses)


@nip
class RobotOnlySampler(AbstractAgentsSampler):

    def __init__(self,
                 sampling_square: Tuple[int, int] = (20, 20),
                 min_robot_goal_distance: float = 5.,
                 max_sample_trials: int = 1000):
        super(RobotOnlySampler, self).__init__(max_peds=0)
        self._sampling_square = sampling_square
        self._min_robot_goal_distance = min_robot_goal_distance
        self._max_sample_trials = max_sample_trials

    def sample(self) -> AgentsSample:
        low = np.array([-self._sampling_square[0] / 2, -self._sampling_square[1] / 2])
        high = np.array([self._sampling_square[0] / 2, self._sampling_square[1] / 2])

        robot_pose = np.random.uniform(low, high, (2,))

        sampled = False
        for _ in range(self._max_sample_trials):
            goal = np.random.uniform(low, high)
            sampled = np.linalg.norm(robot_pose - goal) >= self._min_robot_goal_distance
            if sampled:
                break
        if not sampled:
            raise RuntimeError("Failed to sample goal")

        robot_pose = np.array([robot_pose[0], robot_pose[1], np.random.uniform(-np.pi, np.pi)])

        return AgentsSample(n_peds=0,
                            robot_initial_pose=robot_pose,
                            robot_goal=goal,
                            world_size=self._sampling_square,
                            ped_initial_poses=None)


@nip
class FixedRobotOnlySampler(AbstractAgentsSampler):

    def __init__(self,
                 robot_pose: Tuple[float, float, float],
                 goal_position: Tuple[float, float]):
        assert len(robot_pose) == 3 and len(goal_position) == 2
        super(FixedRobotOnlySampler, self).__init__(max_peds=0)
        self._robot_pose = np.array(robot_pose)
        self._goal_position = np.array(goal_position)

    def sample(self) -> AgentsSample:
        return AgentsSample(n_peds=0,
                            robot_initial_pose=self._robot_pose.copy(),
                            robot_goal=self._goal_position.copy(),
                            world_size=(20, 20),  # TODO: Remove this parameter in sampling system
                            ped_initial_poses=None)
