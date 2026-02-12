import numpy as np
from reservoirpy.datasets import santafe_laser, narma


def narmax(order: int, num_timesteps: int=2000, discard: int=20):
    """
    Creates NARMA-X sequence for t timesteps, where X is the order of the system.
    """
    num_timesteps += discard
    # input
    u = np.random.uniform(0, 0.5, (num_timesteps+order, 1)).astype(np.float64)
    y = narma(n_timesteps=num_timesteps, order=order, u=u)
    # discard transient effects from first 20 steps
    u = u[discard+order:]
    y = y[discard:]
    return u.T, y.T


def santa_fe(num_timesteps: int=2000, discard: int=20):
    """
    Creates NARMA-X sequence for t timesteps, where X is the order of the system.
    """
    num_timesteps += discard
    # input
    u = np.random.uniform(0, 0.5, num_timesteps).astype(np.float64)
    # NARMA sequence
    y = santafe_laser()[:num_timesteps].T

    # discard transient effects from first 20 steps
    u = u[np.newaxis, discard:]
    y = y[:, discard:]

    return u, y


    




