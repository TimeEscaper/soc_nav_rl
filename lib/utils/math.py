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

