import numpy as np

from typing import Union


def unnormalize_symmetric(action: np.ndarray,
                          lb: np.ndarray,
                          ub: np.ndarray) -> np.ndarray:
    deviation = (ub - lb) / 2.
    shift = (ub + lb) / 2.
    action = (action * deviation) + shift
    return action


def normalize_asymmetric(value: Union[float, np.ndarray],
                         lb: Union[float, np.ndarray],
                         ub: Union[float, np.ndarray]):
    return (value - lb) / (ub - lb)


def normalize_symmetric(value: Union[float, np.ndarray],
                        lb: Union[float, np.ndarray],
                        ub: Union[float, np.ndarray]):
    ratio = 2. / (ub - lb)
    shift = (ub + lb) / 2.
    return (value - shift) * ratio


def local_polar_to_global(robot_pose: np.ndarray, point_polar: np.ndarray) -> np.ndarray:
    x_rel_rot = point_polar[0] * np.cos(point_polar[1])
    y_rel_rot = point_polar[0] * np.sin(point_polar[1])
    theta = robot_pose[2]
    x_rel = x_rel_rot * np.cos(theta) - y_rel_rot * np.sin(theta)
    y_rel = x_rel_rot * np.sin(theta) + y_rel_rot * np.cos(theta)
    x_abs = x_rel + robot_pose[0]
    y_abs = y_rel + robot_pose[1]
    return np.array([x_abs, y_abs])
