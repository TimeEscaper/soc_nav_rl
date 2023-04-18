import gym
import numpy as np

from abc import ABC, abstractmethod
from typing import Union, List, Any, Dict, Tuple, Optional, TypeAlias
from pyminisim.core import SimulationState, ROBOT_RADIUS, PEDESTRIAN_RADIUS
from pyminisim.visual import AbstractDrawing
from nip import nip

from lib.envs import ProblemConfig, SimConfig, AbstractCurriculum
from lib.envs.core_env import PyMiniSimCoreEnv
from lib.utils.math import unnormalize_symmetric

ObservationType: TypeAlias = Optional[Union[np.ndarray, Dict[np.ndarray]]]
StepReturnType: TypeAlias = Tuple[ObservationType, Optional[float], bool, Dict[str, Any]]


class AbstractTaskEnv(ABC, gym.Env):

    @abstractmethod
    def step(self, action) -> StepReturnType:
        raise NotImplementedError()

    @abstractmethod
    def reset(self) -> ObservationType:
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


class AbstractTaskWrapper(ABC, AbstractTaskEnv):

    def __init__(self, env: AbstractTaskEnv):
        super(AbstractTaskWrapper, self).__init__()
        self._env = env

    def step(self, action) -> StepReturnType:
        return self._env.step(action)

    def reset(self) -> ObservationType:
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

    def step(self, action: np.ndarray) -> StepReturnType:
        has_collision, goal_reached, min_separation_distance = self._env.step(action)
        done = has_collision or goal_reached
        info = {"min_separation": min_separation_distance}
        if has_collision:
            info["done_reason"] = "collision"
        elif goal_reached:
            info["done_reason"] = "success"
        return None, None, done, info

    def reset(self) -> ObservationType:
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
                                               shape=2,
                                               dtype=dtype)
        else:
            self.action_space = gym.spaces.Box(low=self._lb.copy(),
                                               high=self._ub.copy(),
                                               shape=2,
                                               dtype=dtype)
        self.observation_space = env.observation_space

    def step(self, action) -> StepReturnType:
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action = unnormalize_symmetric(action, self._lb, self._ub)
        return self._env.step(action)


class SARLObservationEnv(AbstractTaskWrapper):

    def __init__(self,
                 env: AbstractTaskEnv,
                 peds_padding: int):
        super(SARLObservationEnv, self).__init__(env)
        self._peds_padding = peds_padding

        self.action_space = env.action_space
        if isinstance(env.observation_space, gym.spaces.Dict):
            obs_dict = env.observation_space.spaces.copy()
        else:
            obs_dict = {}
        obs_dict.update({
            # d_g, phi_goal, v_x, v_y, r
            "robot": gym.spaces.Box(low=np.array([0., -np.pi, -np.inf, -np.inf, 0.]),
                                    high=np.array([np.inf, np.pi, np.inf, np.inf, ROBOT_RADIUS]),
                                    shape=(5,)),
            # p_x, p_y, phi, v_x, v_y, r_ped, d, r_ped + r_robot
            "peds": gym.spaces.Box(low=np.array([-np.inf, -np.inf, -np.pi, -np.inf, -np.inf, 0., 0., 0.]),
                                   high=np.tile(np.array([np.inf, np.inf, np.pi, np.inf, np.inf, PEDESTRIAN_RADIUS,
                                                          np.inf, PEDESTRIAN_RADIUS + ROBOT_RADIUS]),
                                                (peds_padding, 1)),
                                   shape=(peds_padding, 8))
        })
        self.observation_space = gym.spaces.Dict(obs_dict)

    def reset(self):
        obs_dict = self._env.reset()
        if not isinstance(obs_dict, dict):
            obs_dict = {}

    def _build_observation(self):
        sim_state: SimulationState = self._env.get_simulation_state()
        goal: np.ndarray = self._env.get_goal()
        
