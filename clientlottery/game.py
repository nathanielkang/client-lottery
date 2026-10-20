"""Synthetic five-client one-step game. No external data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_CLIENTS = 5
GAME_ID = "sigmoid_latent_v1"

# Roles are fixed with the coefficients below.
# 0 helpful main effect
# 1 near-zero alone, large positive interaction with 0 (complement)
# 2 harmful main effect
# 3 moderate positive main effect
# 4 weak negative main effect
CLIENT_ROLES = (
    "helpful_main",
    "complement_of_0",
    "harmful_main",
    "moderate_positive",
    "weak_negative",
)

MAIN = np.array([1.70, 0.02, -1.80, 0.62, -0.55], dtype=np.float64)
GAMMA_01 = 2.40  # complementarity on the latent, before the logistic
COMPLEMENT_PAIR = (0, 1)

# Known constant inclusion probability. The concentration floor is the
# largest kappa such that kappa <= p <= 1-kappa. Draws are plain Bernoulli(p);
# kappa is not a rejection sampler.
POLICIES = (0.5, 0.3)
DP_SIGMA = 1.0
QUORUM = 3

# Pre-declared paper protocol. Smoke must not be reported as this protocol.
PAPER_T = 4000
PAPER_SEEDS = (0, 1, 2, 3, 4)
SMOKE_T = 80
SMOKE_SEEDS = (0,)

# Gauss-Hermite order for the Gaussian expectation of the logistic.
HERMITE_N = 96

_HERMITE_NODES, _HERMITE_WEIGHTS = np.polynomial.hermite.hermgauss(HERMITE_N)
_HERMITE_PROB = _HERMITE_WEIGHTS / np.sqrt(np.pi)


def logistic(x: np.ndarray | float) -> np.ndarray:
    """Map a latent state into (0, 1). Stable for large |x|."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty(np.shape(x), dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out


U0 = float(logistic(0.0))


def latent_of(z: np.ndarray, intercept: float = 0.0) -> np.ndarray:
    """Deterministic pre-noise aggregate state. z is (..., 5) in {0, 1}."""
    z = np.asarray(z, dtype=np.float64)
    base = z @ MAIN
    inter = GAMMA_01 * z[..., COMPLEMENT_PAIR[0]] * z[..., COMPLEMENT_PAIR[1]]
    return base + inter + float(intercept)


def kappa_of(p: float) -> float:
    p = float(p)
    if not (0.0 < p < 1.0):
        raise ValueError(f"p must lie in (0, 1), got {p}")
    return float(min(p, 1.0 - p))


def policy_name(p: float) -> str:
    if abs(float(p) - 0.5) < 1e-15:
        return "p_half_uniform_banzhaf_of_stated_game"
    return "policy_weighted_not_uniform_banzhaf"


@dataclass(frozen=True)
class Setting:
    name: str
    p: float
    sigma: float
    quorum: int | None
    drift_amplitude: float = 0.0

    @property
    def kappa(self) -> float:
        return kappa_of(self.p)

    @property
    def policy(self) -> str:
        return policy_name(self.p)

    @property
    def quorum_label(self) -> str:
        if self.quorum is None:
            return ""
        return "public_noop_below_quorum_effect_includes_enabling"


# One extra public path. It is not a replacement for the frozen grid.
# A full sine cycle keeps the intercept mean near zero; 0.30 is small next to
# the locked main effects 1.70 and -1.80.
DRIFT_AMPLITUDE = 0.30
DRIFT_SETTING_NAME = "drift_nonprivate"


def slow_intercept(t_rounds: int, amplitude: float) -> np.ndarray:
    """H_t-measurable latent intercept. Does not depend on the current lottery."""
    t_rounds = int(t_rounds)
    if t_rounds < 1:
        raise ValueError("T must be positive")
    if amplitude < 0:
        raise ValueError("amplitude must be non-negative")
    t = np.arange(t_rounds, dtype=np.float64)
    return float(amplitude) * np.sin(2.0 * np.pi * t / float(t_rounds))


def bernoulli_settings() -> tuple[Setting, ...]:
    """Frozen-state grid, plus one slow-intercept path at the same coefficients."""
    return (
        Setting("frozen_nonprivate", p=0.5, sigma=0.0, quorum=None),
        Setting("frozen_nonprivate", p=0.3, sigma=0.0, quorum=None),
        Setting("frozen_dp", p=0.5, sigma=DP_SIGMA, quorum=None),
        Setting("frozen_dp", p=0.3, sigma=DP_SIGMA, quorum=None),
        Setting("frozen_quorum_noop", p=0.5, sigma=0.0, quorum=QUORUM),
        Setting(
            DRIFT_SETTING_NAME,
            p=0.5,
            sigma=0.0,
            quorum=None,
            drift_amplitude=DRIFT_AMPLITUDE,
        ),
    )


def all_coalitions(n: int = N_CLIENTS) -> np.ndarray:
    masks = np.arange(2**n, dtype=np.uint32)
    bits = ((masks[:, None] >> np.arange(n, dtype=np.uint32)) & 1).astype(np.float64)
    return bits


def expected_utility(latent: np.ndarray, sigma: float) -> np.ndarray:
    """E[logistic(w + eta)] for eta ~ N(0, sigma^2). Equals logistic(w) at sigma = 0.

    The logistic is nonlinear, so this expectation is not logistic(w) when sigma > 0.
    """
    latent = np.asarray(latent, dtype=np.float64)
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if sigma == 0.0:
        return logistic(latent)
    scale = float(sigma) * np.sqrt(2.0)
    # nodes integrate e^{-x^2}; Z = sqrt(2) x is standard normal
    shifted = latent[..., None] + scale * _HERMITE_NODES
    return logistic(shifted) @ _HERMITE_PROB


def expected_utility_second_moment(latent: np.ndarray, sigma: float) -> np.ndarray:
    """E[logistic(w + eta)^2]. Used only for the exact variance split at b = 0."""
    latent = np.asarray(latent, dtype=np.float64)
    if sigma == 0.0:
        u = logistic(latent)
        return u * u
    scale = float(sigma) * np.sqrt(2.0)
    shifted = latent[..., None] + scale * _HERMITE_NODES
    u = logistic(shifted)
    return (u * u) @ _HERMITE_PROB


def coalition_mean_delta(z: np.ndarray, sigma: float, quorum: int | None) -> np.ndarray:
    """E[D | Z=z] with D = u(w+eta) - u(0), and D = 0 when a quorum no-op fires.

    Aborted coalitions stay in the expectation. They are not dropped.
    """
    z = np.asarray(z, dtype=np.float64)
    mean_u = expected_utility(latent_of(z), sigma)
    delta = mean_u - U0
    if quorum is not None:
        abort = z.sum(axis=-1) < int(quorum)
        delta = np.where(abort, 0.0, delta)
    return delta


def coalition_var_delta(z: np.ndarray, sigma: float, quorum: int | None) -> np.ndarray:
    """Var(D | Z=z). Zero on a no-op, and zero when sigma = 0."""
    z = np.asarray(z, dtype=np.float64)
    if quorum is not None:
        abort = z.sum(axis=-1) < int(quorum)
    else:
        abort = np.zeros(z.shape[:-1], dtype=bool)
    if sigma == 0.0:
        return np.zeros(z.shape[:-1], dtype=np.float64)
    mean_u = expected_utility(latent_of(z), sigma)
    second = expected_utility_second_moment(latent_of(z), sigma)
    var_u = np.maximum(second - mean_u * mean_u, 0.0)
    return np.where(abort, 0.0, var_u)


def draw_round(
    rng: np.random.Generator,
    T: int,
    p: float,
    sigma: float,
    quorum: int | None,
    intercept: np.ndarray | float = 0.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """One frozen-state transcript of T independent rounds.

    Returns participation Z (T, 5), utility change D (T,), and the number of
    quorum aborts included in D. Aborted rounds are kept with D = 0.
    """
    if T < 1:
        raise ValueError("T must be positive")
    z = (rng.random((T, N_CLIENTS)) < float(p)).astype(np.float64)
    latent = latent_of(z)
    if sigma > 0.0:
        eta = rng.normal(0.0, float(sigma), size=T)
    else:
        eta = np.zeros(T, dtype=np.float64)
    # Noise hits the state before the nonlinear utility.
    d = logistic(latent + eta + np.asarray(intercept, dtype=np.float64)) - U0
    n_aborted = 0
    if quorum is not None:
        abort = z.sum(axis=1) < int(quorum)
        n_aborted = int(abort.sum())
        d = np.where(abort, 0.0, d)
    return z, d, n_aborted


def draw_fixed_size(
    rng: np.random.Generator,
    T: int,
    k: int,
    sigma: float,
) -> np.ndarray:
    """Exactly k clients each round, uniformly among combinations. No quorum."""
    if not (1 <= k < N_CLIENTS):
        raise ValueError(f"k must lie in 1..{N_CLIENTS - 1}")
    # Enumerate combinations once.
    combos = []
    # 5 choose k is at most 10
    from itertools import combinations

    for combo in combinations(range(N_CLIENTS), k):
        row = np.zeros(N_CLIENTS, dtype=np.float64)
        row[list(combo)] = 1.0
        combos.append(row)
    table = np.vstack(combos)
    idx = rng.integers(0, table.shape[0], size=T)
    return table[idx]


def fixed_size_outcomes(z: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    latent = latent_of(z)
    if sigma > 0.0:
        eta = rng.normal(0.0, float(sigma), size=z.shape[0])
    else:
        eta = 0.0
    return logistic(latent + eta) - U0
