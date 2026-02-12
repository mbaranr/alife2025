import numpy as np
from grow.reservoir import Reservoir


def spectral_radius(res: Reservoir):
    eigenvalues = np.linalg.eigvals(res.A)  
    return max(abs(eigenvalues))


def _effective_rank(singular_values: np.ndarray, threshold: float = 0.99) -> int:
    """
    Computes the number of singular values required to 
    capture the specified percentage of the total sum.
    """
    full_sum = np.sum(singular_values)
    cutoff = threshold * full_sum
    tmp_sum = 0.0
    e_rank = 0

    for val in singular_values:
        tmp_sum += val
        e_rank += 1
        if tmp_sum >= cutoff:
            break
    return e_rank


def kernel_rank(res: Reservoir, 
                num_timesteps: int=2000):
    """
    Computes the number of non-zero 
    singular values of the reservoir state matrix.
    """
    # generate random input signal in [-1, 1]
    ui = 2 * np.random.rand(num_timesteps) - 1
    input = np.tile(ui[:, None], (1, res.input_units)).T.astype(np.float64)

    res.reset()
    _ = res.run(input)

    state = res.reservoir_state[:, res.washout:]
    s = np.linalg.svd(state, compute_uv=False)
    return _effective_rank(s)
    

def generalization_rank(res: Reservoir, 
                        num_timesteps: int=2000):
    """
    Computes the Magnitude Generalizationr Rank (MGR).
    """
    # generate random input signal in [0.45, 0.55]
    input = 0.5 + 0.1 * np.random.rand(res.input_units, num_timesteps) - 0.05

    res.reset()
    _ = res.run(input)

    state = res.reservoir_state[:, res.washout:]
    s = np.linalg.svd(state, compute_uv=False)
    return _effective_rank(s)

###### memory capacity ######

def _generate_input_signal(num_timesteps, max_delay):
    total_length = num_timesteps + max_delay + 1
    return 2 * np.random.rand(1, total_length) - 1

def _delayed_targets(input_signal, num_timesteps, max_delay):
    target_delays = np.zeros((num_timesteps, max_delay))
    for delay in range(1, max_delay + 1):
        target_delays[:, delay - 1] = input_signal[:, max_delay - delay: max_delay + num_timesteps - delay].T[:, 0]
    return target_delays

def _compute_mc(y_true, y_pred, filter=0.1):
    cov = np.cov(y_true, y_pred, ddof=1)[0, 1]
    var_pred = np.var(y_pred)
    var_true = np.var(y_true)
    denom = var_true * var_pred
    mc = (cov ** 2) / denom if denom != 0 else 0.0
    return mc if mc > filter else 0.0


def linear_memory_capacity(res: Reservoir,
                           num_timesteps=2000,
                           max_delay=None,
                           filter=0.1,
                           normalize=True):
    if not max_delay:
        max_delay = res.size()

    sequence_length = num_timesteps // 2
    input_signal = _generate_input_signal(num_timesteps, max_delay)
    input_sequence = input_signal[:, max_delay:max_delay + num_timesteps].T
    targets = _delayed_targets(input_signal, num_timesteps, max_delay)

    # train/test split
    train_input = input_sequence[:sequence_length].T
    test_input = input_sequence[sequence_length:].T
    train_target = targets[:sequence_length].T
    test_target = targets[sequence_length:]
    test_target = test_target[res.washout:, :]

    res.reset()
    res.train(train_input, train_target)
    predictions = res.run(test_input)

    mcs = [_compute_mc(test_target[:, i], predictions[i, :], filter) for i in range(max_delay)]
    return np.mean(mcs) if normalize else np.sum(mcs)


def cross_memory_capacity(res: Reservoir,
                          num_timesteps=2000,
                          max_delay=30,
                          filter=0.1,
                          normalize=True):
    if not max_delay:
        max_delay = res.size()

    sequence_length = num_timesteps // 2
    input_signal = _generate_input_signal(num_timesteps, max_delay)
    input_sequence = input_signal[:, max_delay:max_delay + num_timesteps].T

    # delayed matrix
    delayed = np.zeros((num_timesteps, max_delay))
    for delay in range(1, max_delay + 1):
        delayed[:, delay - 1] = input_signal[:, max_delay - delay: max_delay + num_timesteps - delay].T[:, 0]

    # cross terms using upper triangle of delayed x delayed
    delayed_ = delayed[:, :, np.newaxis]
    products = delayed_ * delayed_[:, np.newaxis, :]
    triu_indices = np.triu_indices(max_delay)
    products_flat = products.reshape(products.shape[0], -1)
    flat_indices = triu_indices[0] * max_delay + triu_indices[1]
    cross_terms = products_flat[:, flat_indices]

    train_input = input_sequence[:sequence_length].T
    test_input = input_sequence[sequence_length:].T
    train_target = cross_terms[:sequence_length].T
    test_target = cross_terms[sequence_length:]
    test_target = test_target[res.washout:, :]

    res.reset()
    res.train(train_input, train_target)
    predictions = res.run(test_input)

    mcs = [_compute_mc(test_target[:, i], predictions[i, :], filter) for i in range(predictions.shape[0])]
    return np.mean(mcs) if normalize else np.sum(mcs)


def quadratic_memory_capacity(res: Reservoir,
                               num_timesteps=2000,
                               max_delay=None,
                               filter=0.1,
                               normalize=True):
    if not max_delay:
        max_delay = res.size()

    sequence_length = num_timesteps // 2
    input_signal = _generate_input_signal(num_timesteps, max_delay)
    input_sequence = input_signal[:, max_delay:max_delay + num_timesteps].T

    # quadratic targets
    targets = _delayed_targets(input_signal, num_timesteps, max_delay)
    targets = targets ** 2

    train_input = input_sequence[:sequence_length].T
    test_input = input_sequence[sequence_length:].T
    train_target = targets[:sequence_length].T
    test_target = targets[sequence_length:]
    test_target = test_target[res.washout:, :]

    res.reset()
    res.train(train_input, train_target)
    predictions = res.run(test_input)

    mcs = [_compute_mc(test_target[:, i], predictions[i, :], filter) for i in range(max_delay)]
    return np.mean(mcs) if normalize else np.sum(mcs)