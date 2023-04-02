import numpy as np


def unnormalize(action: np.ndarray,
                lb: np.ndarray,
                ub: np.ndarray) -> np.ndarray:
    deviation = (ub - lb) / 2.
    shift = (ub + lb) / 2.
    action = (action * deviation) + shift
    return action
