"""Estimators and the deliberately unidentified round-fixed-effect diagnostic."""

from __future__ import annotations

import numpy as np

from .game import N_CLIENTS


def expanding_baseline(d: np.ndarray) -> np.ndarray:
    """Forecast of D_t from previous releases only. b_0 = 0."""
    d = np.asarray(d, dtype=np.float64)
    b = np.zeros(d.shape[0], dtype=np.float64)
    if d.shape[0] > 1:
        csum = np.cumsum(d)
        b[1:] = csum[:-1] / np.arange(1, d.shape[0], dtype=np.float64)
    return b


def ipw_weights(z: np.ndarray, p: float) -> np.ndarray:
    """A_it = (Z_it - p) / (p (1-p)). p is known."""
    p = float(p)
    return (z - p) / (p * (1.0 - p))


def ipw_estimates(z: np.ndarray, d: np.ndarray, p: float, baseline: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (tauhat, psi) with psi shaped (T, n)."""
    if baseline == "zero":
        b = np.zeros(d.shape[0], dtype=np.float64)
    elif baseline == "expanding_mean_past_release":
        b = expanding_baseline(d)
    else:
        raise ValueError(baseline)
    psi = ipw_weights(z, p) * (d - b)[:, None]
    return psi.mean(axis=0), psi


def inclusion_exclusion(z: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Plain mean(D | Z_i=1) - mean(D | Z_i=0). Blank where a cell is empty."""
    est = np.full(z.shape[1], np.nan, dtype=np.float64)
    varied = np.zeros(z.shape[1], dtype=bool)
    for i in range(z.shape[1]):
        inc = z[:, i] == 1.0
        exc = ~inc
        varied[i] = bool(inc.any() and exc.any())
        if varied[i]:
            est[i] = d[inc].mean() - d[exc].mean()
    return est, varied


def linear_regression(z: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, int, int, bool]:
    """OLS of D on an intercept and the participation columns.

    When those columns are independent Bernoulli draws, the population
    coefficient on client i equals the participation effect for any coalition
    utility. The coefficient does not recover the full coalition function.
    Returns nan coefficients when the transcript is rank-deficient.
    """
    x = np.column_stack([np.ones(z.shape[0], dtype=np.float64), z])
    rank = int(np.linalg.matrix_rank(x, tol=1e-8))
    ncols = int(x.shape[1])
    identified = rank == ncols
    est = np.full(z.shape[1], np.nan, dtype=np.float64)
    if identified:
        coef, _, _, _ = np.linalg.lstsq(x, d, rcond=None)
        est[:] = coef[1:]
    return est, rank, ncols, identified


def round_fixed_effect_diagnostic(z: np.ndarray) -> dict:
    """Saturated round effects with one utility change per round.

    Each round is its own group. The within-round mean of a participation
    column is that single observation, so the residual is zero and the client
    Gram matrix is the zero matrix. Rank of [round dummies | participation]
    equals the number of rounds. Client coefficients are unidentified.
    This function does not return a coefficient vector.
    """
    z = np.asarray(z, dtype=np.float64)
    t_rounds, n_clients = z.shape
    groups = np.arange(t_rounds)
    resid = np.empty_like(z)
    for i in range(n_clients):
        sums = np.bincount(groups, weights=z[:, i], minlength=t_rounds)
        counts = np.bincount(groups, minlength=t_rounds).astype(np.float64)
        resid[:, i] = z[:, i] - sums / counts
    gram = resid.T @ resid
    gram_max = float(np.max(np.abs(gram)))
    resid_max = float(np.max(np.abs(resid)))
    rank_check = "algebraic_one_observation_per_round"
    design_rank = int(t_rounds)
    if t_rounds <= 256:
        eye = np.eye(t_rounds, dtype=np.float64)
        design = np.column_stack([eye, z])
        design_rank = int(np.linalg.matrix_rank(design, tol=1e-8))
        rank_check = "numerical_svd"
        if design_rank != t_rounds:
            raise RuntimeError(
                f"saturated design rank {design_rank} was not T={t_rounds}"
            )
    return {
        "identified": False,
        "failure_reason": "saturated_round_fixed_effects_residual_participation_is_zero",
        "gram_max_abs": gram_max,
        "residual_max_abs": resid_max,
        "design_rank": design_rank,
        "design_ncols": int(t_rounds + n_clients),
        "rank_check": rank_check,
        "n_clients": int(n_clients),
    }


def freedman_halfwidth(t_rounds: int, kappa: float, delta: float) -> float:
    """Note's conservative radius for the average of psi.

    v = T * (4/kappa)^2 bounds the sum of conditional second moments because
    |psi - tau| <= 4/kappa when |D| <= 1 and |b| <= 1. This is not a
    sample-variance plug-in and it is not called a certificate.
    """
    if kappa <= 0 or delta <= 0 or delta >= 1:
        raise ValueError("kappa and delta out of range")
    bound = 4.0 / float(kappa)
    v = float(t_rounds) * (bound ** 2)
    log_term = float(np.log(2.0 / float(delta)))
    radius = np.sqrt(2.0 * v * log_term) + (2.0 / 3.0) * bound * log_term
    return float(radius / float(t_rounds))


def plugin_halfwidth_not_certified(psi_i: np.ndarray, delta: float, kappa: float) -> float:
    """Same algebra as the conservative radius with v = T * sample variance.

    Not a certificate. Do not plug an unverified sample variance into v
    and call the result certified.
    """
    psi_i = np.asarray(psi_i, dtype=np.float64)
    t_rounds = int(psi_i.shape[0])
    if t_rounds < 2:
        return float("nan")
    sample_var = float(np.var(psi_i, ddof=1))
    bound = 4.0 / float(kappa)
    v = float(t_rounds) * sample_var
    log_term = float(np.log(2.0 / float(delta)))
    radius = np.sqrt(2.0 * v * log_term) + (2.0 / 3.0) * bound * log_term
    return float(radius / float(t_rounds))


def sign_of(value: float, tol: float = 1e-12) -> int:
    if not np.isfinite(value):
        return 0
    if value > tol:
        return 1
    if value < -tol:
        return -1
    return 0


def ranks_descending(values: np.ndarray) -> np.ndarray:
    """1 = largest. Ties share the minimum rank in the tied block."""
    values = np.asarray(values, dtype=np.float64)
    ranks = np.full(values.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(values)
    idx = np.flatnonzero(finite)
    if idx.size == 0:
        return ranks
    order = idx[np.argsort(-values[idx], kind="mergesort")]
    i = 0
    while i < order.size:
        j = i
        while j + 1 < order.size and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = float(i + 1)
        i = j + 1
    return ranks


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    i = 0
    n = values.shape[0]
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = 0.5 * ((i + 1) + (j + 1))
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return float("nan")
    ra = average_ranks(a[mask])
    rb = average_ranks(b[mask])
    if float(np.std(ra)) < 1e-15 or float(np.std(rb)) < 1e-15:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def mean_abs_error(estimate: np.ndarray, oracle: np.ndarray) -> float:
    mask = np.isfinite(estimate) & np.isfinite(oracle)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(estimate[mask] - oracle[mask])))
