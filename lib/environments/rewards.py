import numpy as np

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from nip import nip


class RewardContext:

    def __init__(self, attributes: Dict[str, Any]):
        self._attributes = attributes

    def has_attribute(self, attribute: str) -> bool:
        return attribute in self._attributes

    def get(self, attribute: str) -> Optional[Any]:
        if attribute in self._attributes:
            return self._attributes[attribute]
        return None


class AbstractReward(ABC):

    @abstractmethod
    def __call__(self, context: RewardContext) -> float:
        raise NotImplementedError()


@nip
class CompositeReward(AbstractReward):

    def __init__(self, rewards: Tuple[AbstractReward], weights: Optional[Tuple[float]] = None):
        super(CompositeReward, self).__init__()
        if weights is not None:
            assert len(rewards) == len(weights), "Number of weights must be same as number of rewards"
        else:
            weights = (1. for _ in range(len(rewards)))
        self._rewards = rewards
        self._weights = weights

    def __call__(self, context: RewardContext) -> float:
        return sum([weight * reward(context) for reward, weight in zip(self._rewards, self._weights)])

