"""Oracles for the missing-experiment battery. No transcripts are drawn here."""

from __future__ import annotations

from math import factorial

import numpy as np

from .game import MAIN, N_CLIENTS
from .missing_spec import (
    DROPOUT_P,
    DROPOUT_RATES,
    ELIGIBILITY_CLIENT,
    PHASE_INTERCEPTS,
    PHASE_SCALES,
)
from .oracle import (
    effects_from_product,
    itt_and_received,
    naive_receipt_population_limit,
    release_means,
)


def shapley_values(values: np.ndarray) -> np.ndarray:
    """Exact Shapley values of the 32-coalition release game."""
    values = np.asarray(values, dtype=np.float64)
    phi = np.zeros(N_CLIENTS, dtype=np.float64)
    for i in range(N_CLIENTS):
        others = [j for j in range(N_CLIENTS) if j != i]
        for mask in range(1 << len(others)):
            size = 0
            base = 0
            for bit_pos, j in enumerate(others):
                if (mask >> bit_pos) & 1:
                    base |= 1 << j
                    size += 1
            weight = factorial(size) * factorial(N_CLIENTS - size - 1) / factorial(N_CLIENTS)
            phi[i] += weight * (float(values[base | (1 << i)]) - float(values[base]))
    return phi


def leave_one_out(values: np.ndarray) -> np.ndarray:
    """v(full) - v(full without i). Six coalition utilities."""
    values = np.asarray(values, dtype=np.float64)
    full = (1 << N_CLIENTS) - 1
    out = np.zeros(N_CLIENTS, dtype=np.float64)
    for i in range(N_CLIENTS):
        out[i] = float(values[full] - values[full ^ (1 << i)])
    return out


def add_one_from_empty(values: np.ndarray) -> np.ndarray:
    """v({i}) - v(empty). Six coalition utilities."""
    values = np.asarray(values, dtype=np.float64)
    out = np.zeros(N_CLIENTS, dtype=np.float64)
    for i in range(N_CLIENTS):
        out[i] = float(values[1 << i] - values[0])
    return out


def phase_bounds(t_rounds: int) -> tuple[int, int]:
    b1 = int(t_rounds) // 3
    b2 = (2 * int(t_rounds)) // 3
    if not (0 < b1 < b2 < int(t_rounds)):
        raise ValueError(f"T={t_rounds} is too small for three phases")
    return b1, b2


def phase_weights(t_rounds: int) -> np.ndarray:
    b1, b2 = phase_bounds(t_rounds)
    lengths = np.array([b1, b2 - b1, int(t_rounds) - b2], dtype=np.float64)
    return lengths / float(t_rounds)


def phase_release_means(phase: int, sigma: float = 0.0) -> np.ndarray:
    scale = float(PHASE_SCALES[phase])
    intercept = float(PHASE_INTERCEPTS[phase])
    return release_means(main=scale * MAIN, intercept=intercept, sigma=sigma)


def drift_strong_targets(t_rounds: int, p: float = 0.5) -> dict:
    """Path-average, last-regime, and one-step change bound for drift_strong."""
    probs = np.full(N_CLIENTS, float(p))
    taus = [effects_from_product(phase_release_means(phase), probs) for phase in range(3)]
    weights = phase_weights(t_rounds)
    path = sum(float(weights[k]) * taus[k] for k in range(3))
    last = taus[2]
    one_step = np.maximum(np.abs(taus[1] - taus[0]), np.abs(taus[2] - taus[1]))
    mean_abs_dev = float(weights[0]) * np.abs(taus[0] - last) + float(weights[1]) * np.abs(taus[1] - last)
    return {
        "phase_tau": taus,
        "path_average": path,
        "last_state": last,
        "trailing_oracle": last.copy(),
        "max_one_step_L": one_step,
        "mean_abs_dev_from_last": mean_abs_dev,
        "weights": weights,
    }


def eligibility_targets(t_rounds: int, p: float = 0.5) -> dict:
    """Frozen coefficients. Middle phase forces one client out of the lottery.

    That client's path target is zero on the ineligible block. The other
    clients' middle-phase target is the add-one contrast with that client
    held out. This is not the drift_strong path.
    """
    values = release_means()
    probs_on = np.full(N_CLIENTS, float(p))
    probs_off = probs_on.copy()
    probs_off[ELIGIBILITY_CLIENT] = 0.0
    tau_on = effects_from_product(values, probs_on)
    tau_off = effects_from_product(values, probs_off)
    tau_off = np.array(tau_off, copy=True)
    tau_off[ELIGIBILITY_CLIENT] = 0.0
    weights = phase_weights(t_rounds)
    path = float(weights[0]) * tau_on + float(weights[1]) * tau_off + float(weights[2]) * tau_on
    return {
        "eligible_tau": tau_on,
        "ineligible_block_tau": tau_off,
        "path_average": path,
        "last_state": tau_on.copy(),
        "weights": weights,
    }


def dropout_population(p: float = DROPOUT_P, rates: np.ndarray | None = None) -> dict:
    if rates is None:
        rates = DROPOUT_RATES
    values = release_means()
    itt, received = itt_and_received(p, rates, values)
    naive = naive_receipt_population_limit(p, rates, values)
    return {
        "values": values,
        "itt": itt,
        "received": received,
        "naive_population_limit": naive,
        "compliance": 1.0 - np.asarray(rates, dtype=np.float64),
    }


def delta_method_se(var_of_mean: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """SE of the product-measure add-one contrast from cell-mean variances."""
    var_of_mean = np.asarray(var_of_mean, dtype=np.float64)
    probs = np.asarray(probs, dtype=np.float64)
    n_clients = int(probs.shape[0])
    se2 = np.zeros(n_clients, dtype=np.float64)
    for i in range(n_clients):
        others = [j for j in range(n_clients) if j != i]
        acc = 0.0
        for mask in range(1 << len(others)):
            weight = 1.0
            idx1 = 1 << i
            idx0 = 0
            for bit_pos, j in enumerate(others):
                bit = (mask >> bit_pos) & 1
                if bit:
                    idx1 |= 1 << j
                    idx0 |= 1 << j
                    weight *= float(probs[j])
                else:
                    weight *= 1.0 - float(probs[j])
            acc += (weight ** 2) * (float(var_of_mean[idx1]) + float(var_of_mean[idx0]))
        se2[i] = acc
    return np.sqrt(se2)


def saturated_reuse(z: np.ndarray, d: np.ndarray, p: float) -> tuple[np.ndarray, np.ndarray, int]:
    """Participation effects from the coalition means in one transcript.

    Uses only (Z_t, D_t). Extra coalition queries: 0. If any of the 32
    coalitions is missing, the known-p plug-in is left blank.
    """
    z = np.asarray(z, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    n_clients = int(z.shape[1])
    n_cells = 1 << n_clients
    idx = np.sum(z.astype(np.int64) * (1 << np.arange(n_clients)), axis=1)
    counts = np.bincount(idx, minlength=n_cells).astype(np.float64)
    sum_d = np.bincount(idx, weights=d, minlength=n_cells)
    sum_d2 = np.bincount(idx, weights=d * d, minlength=n_cells)
    n_obs = int(np.sum(counts > 0))
    nan = np.full(n_clients, np.nan, dtype=np.float64)
    if np.any(counts <= 0.0):
        return nan, nan, n_obs
    means = sum_d / counts
    var_mean = np.zeros(n_cells, dtype=np.float64)
    multi = counts >= 2.0
    ss = sum_d2[multi] - (sum_d[multi] ** 2) / counts[multi]
    sample_var = np.maximum(ss / (counts[multi] - 1.0), 0.0)
    var_mean[multi] = sample_var / counts[multi]
    probs = np.full(n_clients, float(p))
    tau = effects_from_product(means, probs)
    se = delta_method_se(var_mean, probs)
    return tau, se, n_obs
