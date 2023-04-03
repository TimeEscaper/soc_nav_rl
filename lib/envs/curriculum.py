from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union, List, Any
from nip import nip

from lib.envs.agents_samplers import AbstractAgentsSampler
from lib.envs.sim_config_samplers import AbstractProblemConfigSampler


class AbstractCurriculum(ABC):

    @abstractmethod
    def get_problem_sampler(self) -> AbstractProblemConfigSampler:
        raise NotImplementedError()

    @abstractmethod
    def get_agents_sampler(self) -> AbstractAgentsSampler:
        raise NotImplementedError()

    @abstractmethod
    def get_success_rate_threshold(self) -> Optional[float]:
        raise NotImplementedError()

    @abstractmethod
    def get_current_stage(self) -> Tuple[int, Optional[str]]:
        raise NotImplementedError()

    @abstractmethod
    def update_stage(self) -> None:
        raise NotImplementedError()


@nip
class DummyCurriculum(AbstractCurriculum):

    def __init__(self,
                 agents_sampler: AbstractAgentsSampler,
                 problem_sampler: AbstractProblemConfigSampler):
        super(DummyCurriculum, self).__init__()
        self._agents_sampler = agents_sampler
        self._problem_sampler = problem_sampler

    def get_problem_sampler(self) -> AbstractProblemConfigSampler:
        return self._problem_sampler

    def get_agents_sampler(self) -> AbstractAgentsSampler:
        return self._agents_sampler

    def get_success_rate_threshold(self) -> Optional[float]:
        return None

    def get_current_stage(self) -> Tuple[int, Optional[str]]:
        return 0, None

    def update_stage(self) -> None:
        pass


@nip
class SequentialCurriculum(AbstractCurriculum):

    def __init__(self,
                 agents_samplers: Union[AbstractAgentsSampler, List[AbstractAgentsSampler]],
                 problem_samplers: Union[AbstractAgentsSampler, List[AbstractProblemConfigSampler]],
                 stages: List[Tuple[str, float]]):
        super(SequentialCurriculum, self).__init__()

        n_stages = len(stages)
        if isinstance(agents_samplers, list):
            assert len(agents_samplers) == n_stages, \
                f"If agents_samplers is a list, it must have length equal to the number of stages which is " \
                f"{n_stages}, length {len(agents_samplers)} is given"
        if isinstance(len(problem_samplers), list):
            assert len(problem_samplers) == n_stages, \
                f"If problem_samplers is a list, it must have length equal to the number of stages which is " \
                f"{n_stages}, length {len(problem_samplers)} is given"

        self._agents_samplers = agents_samplers
        self._problem_samplers = problem_samplers
        self._stages = stages
        self._current_stage_idx = 0
        self._n_stages = n_stages
        self._is_steady = False

    def get_problem_sampler(self) -> AbstractProblemConfigSampler:
        return SequentialCurriculum._get_item(self._problem_samplers, self._current_stage_idx)

    def get_agents_sampler(self) -> AbstractAgentsSampler:
        return SequentialCurriculum._get_item(self._agents_samplers, self._current_stage_idx)

    def get_success_rate_threshold(self) -> Optional[float]:
        if not self._is_steady:
            return self._stages[self._current_stage_idx][1]
        return None

    def get_current_stage(self) -> Tuple[int, Optional[str]]:
        if not self._is_steady:
            return self._current_stage_idx, self._stages[self._current_stage_idx][0]
        return self._current_stage_idx + 1, None

    def update_stage(self) -> None:
        new_stage_idx = self._current_stage_idx + 1
        if new_stage_idx >= self._n_stages:
            self._is_steady = True
        else:
            self._current_stage_idx = new_stage_idx

    @staticmethod
    def _get_item(array: Union[Any, List[Any]], idx: int) -> Any:
        if isinstance(array, list):
            return array[idx]
        return array
