import random
import numpy as np
import torch

from typing import Optional, Union, Any, Tuple, List, Callable
from scipy.spatial.distance import cdist


def get_or_sample_uniform(value: Optional[Union[Any, Tuple[Any, Any]]],
                          size: Optional[Union[int, Tuple[int, ...]]] = None) -> Any:
    if value is None:
        return None
    if isinstance(value, tuple):
        return np.random.uniform(value[0], value[1], size=size)
    if size is None:
        return value
    return np.repeat(value, size)


def get_or_sample_choice(value: Optional[Union[Any, Tuple[Any, ...], List[Any]]]) -> Any:
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


def get_or_sample_int(value: Union[int, Tuple[int, int]]) -> int:
    if isinstance(value, tuple):
        return np.random.randint(value[0], value[1] + 1)
    return value


def random_angle(size: Optional[Union[int, Tuple[int, ...]]] = None):
    return np.random.uniform(-np.pi, np.pi, size)


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_joint_positions(sampling_fn: Callable[..., np.ndarray], distance_threshold: float,
                           max_trials: int = 100) -> np.ndarray:
    sampled = False
    positions = None
    for _ in range(max_trials):
        positions = sampling_fn()
        dists = cdist(positions, positions, "euclidean")
        dists = dists[np.triu_indices(dists.shape[0], 1)]
        sampled = (dists > distance_threshold).all()
        if sampled:
            break
    if not sampled:
        raise RuntimeError("Failed to sample positions")
    return positions


def sample_joint_positions_uniform(low: np.ndarray, high: np.ndarray, n: int, distance_threshold: float,
                                   max_trials: int = 100) -> np.ndarray:
    return sample_joint_positions(lambda: np.random.uniform(low, high, (n, 2)),
                                  distance_threshold, max_trials)


def sample_goal(sampling_fn: Callable[..., np.ndarray], initial_pose: np.ndarray, min_distance: float,
                max_trials: int = 100) -> np.ndarray:
    sampled = False
    goal = None
    for _ in range(max_trials):
        goal = sampling_fn()
        sampled = np.linalg.norm(initial_pose - goal) >= min_distance
        if sampled:
            break
    if not sampled:
        raise RuntimeError("Failed to sample goal")
    return goal


def sample_goal_uniform(low: np.ndarray, high: np.ndarray, initial_pose: np.ndarray, min_distance: float,
                        max_trials: int = 100) -> np.ndarray:
    return sample_goal(lambda: np.random.uniform(low, high), initial_pose, min_distance, max_trials)
