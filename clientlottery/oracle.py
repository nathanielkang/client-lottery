"""Exact policy-weighted participation effects by coalition enumeration."""

from __future__ import annotations

import numpy as np

from .game import (
    N_CLIENTS,
    U0,
    all_coalitions,
    coalition_mean_delta,
    coalition_var_delta,
    expected_utility,
    latent_of,
)


def participation_effects(
    p: float,
    sigma: float,
    quorum: int | None,
) -> np.ndarray:
    """tau_i = E[m(1, Z_{-i}) - m(0, Z_{-i})] with Z_{-i} ~ Bernoulli(p)^{n-1}.

    Five clients give 16 coalitions of the others. The inner expectation over
    Gaussian state noise, when sigma > 0, is inside m. At p = 1/2 this is the
    uniform Banzhaf marginal of the stated game. At any other p it is the
    policy-weighted marginal of that same game.
    """
    p = float(p)
    coalitions = all_coalitions(N_CLIENTS)
    means = coalition_mean_delta(coalitions, sigma=sigma, quorum=quorum)
    tau = np.zeros(N_CLIENTS, dtype=np.float64)
    for i in range(N_CLIENTS):
        acc = 0.0
        # Loop over the 16 assignments of the other clients.
        for mask in range(2 ** (N_CLIENTS - 1)):
            z1 = np.zeros(N_CLIENTS, dtype=np.float64)
            z0 = np.zeros(N_CLIENTS, dtype=np.float64)
            weight = 1.0
            bit_pos = 0
            for j in range(N_CLIENTS):
                if j == i:
                    continue
                bit = (mask >> bit_pos) & 1
                bit_pos += 1
                z1[j] = bit
                z0[j] = bit
                weight *= p if bit else (1.0 - p)
            z1[i] = 1.0
            z0[i] = 0.0
            # Look up the precomputed coalition means. Bit j of the index is client j.
            idx1 = int(np.round(z1) @ (1 << np.arange(N_CLIENTS)))
            idx0 = int(np.round(z0) @ (1 << np.arange(N_CLIENTS)))
            acc += weight * (means[idx1] - means[idx0])
        tau[i] = acc
    return tau


def path_average_participation_effects(
    p: float,
    sigma: float,
    quorum: int | None,
    intercepts: np.ndarray,
) -> np.ndarray:
    """Average of tau_i(c_t) along a public intercept path.

    The target is the window average of the current-round effect, which is
    what the IPW average estimates when the intercept is H_t-measurable.
    """
    p = float(p)
    intercepts = np.asarray(intercepts, dtype=np.float64)
    coalitions = all_coalitions(N_CLIENTS)
    base = latent_of(coalitions)
    ones = coalitions.sum(axis=1)
    prob = (p ** ones) * ((1.0 - p) ** (float(N_CLIENTS) - ones))
    acc = np.zeros(N_CLIENTS, dtype=np.float64)
    t_rounds = int(intercepts.shape[0])
    for start in range(0, t_rounds, 8192):
        chunk = intercepts[start : start + 8192]
        latent = base[:, None] + chunk[None, :]
        means = expected_utility(latent, sigma) - U0
        if quorum is not None:
            abort = coalitions.sum(axis=1) < int(quorum)
            means = np.array(means, copy=True)
            means[abort, :] = 0.0
        for i in range(N_CLIENTS):
            inn = coalitions[:, i] == 1.0
            m1 = np.sum(prob[inn, None] * means[inn], axis=0) / p
            m0 = np.sum(prob[~inn, None] * means[~inn], axis=0) / (1.0 - p)
            acc[i] += float(np.sum(m1 - m0))
    return acc / float(t_rounds)


def contrast_from_means(coalitions: np.ndarray, means: np.ndarray, p: float) -> np.ndarray:
    """E[m|Z_i=1] - E[m|Z_i=0] from a complete table of coalition means."""
    p = float(p)
    ones = coalitions.sum(axis=1)
    prob = (p ** ones) * ((1.0 - p) ** (coalitions.shape[1] - ones))
    tau = np.zeros(coalitions.shape[1], dtype=np.float64)
    for i in range(coalitions.shape[1]):
        in_i = coalitions[:, i] == 1.0
        m1 = np.sum(prob[in_i] * means[in_i]) / p
        m0 = np.sum(prob[~in_i] * means[~in_i]) / (1.0 - p)
        tau[i] = m1 - m0
    return tau


def conditional_mean_contrast(p: float, sigma: float, quorum: int | None) -> np.ndarray:
    """Same estimand via E[m|Z_i=1] - E[m|Z_i=0], enumerating all 32 coalitions."""
    coalitions = all_coalitions(N_CLIENTS)
    means = coalition_mean_delta(coalitions, sigma=sigma, quorum=quorum)
    return contrast_from_means(coalitions, means, p)


def partner_conditional_effect(
    client: int,
    partner: int,
    partner_value: int,
    p: float,
    sigma: float,
) -> float:
    """Policy-weighted add-one effect of `client` with `partner` held fixed.

    The remaining clients stay Bernoulli(p). Used to show complementarity;
    it is not the marginal estimand.
    """
    others = [j for j in range(N_CLIENTS) if j != client and j != partner]
    acc = 0.0
    for mask in range(2 ** len(others)):
        z1 = np.zeros(N_CLIENTS, dtype=np.float64)
        z0 = np.zeros(N_CLIENTS, dtype=np.float64)
        weight = 1.0
        for b, j in enumerate(others):
            bit = (mask >> b) & 1
            z1[j] = bit
            z0[j] = bit
            weight *= p if bit else (1.0 - p)
        z1[partner] = partner_value
        z0[partner] = partner_value
        z1[client] = 1.0
        z0[client] = 0.0
        d1 = float(coalition_mean_delta(z1, sigma=sigma, quorum=None))
        d0 = float(coalition_mean_delta(z0, sigma=sigma, quorum=None))
        acc += weight * (d1 - d0)
    return float(acc)


def exact_psi_variance_b0(p: float, sigma: float, quorum: int | None) -> dict:
    """Population variance split of psi = A (D - 0) at a frozen state.

    Var(psi_i) = Var_Z[A_i (m(Z))] + E_Z[A_i(Z)^2 Var(D|Z)].
    The first term is design variance. The second is utility variance after
    the nonlinear map of Gaussian state noise. Neither term is the gap
    between the DP game and the nonprivate game.
    """
    p = float(p)
    coalitions = all_coalitions(N_CLIENTS)
    means = coalition_mean_delta(coalitions, sigma=sigma, quorum=quorum)
    vars_ = coalition_var_delta(coalitions, sigma=sigma, quorum=quorum)
    ones = coalitions.sum(axis=1)
    prob = (p ** ones) * ((1.0 - p) ** (N_CLIENTS - ones))
    design = np.zeros(N_CLIENTS, dtype=np.float64)
    noise = np.zeros(N_CLIENTS, dtype=np.float64)
    for i in range(N_CLIENTS):
        a = np.where(
            coalitions[:, i] == 1.0,
            1.0 / p,
            -1.0 / (1.0 - p),
        )
        score = a * means
        # E[A]=0 and E[A m] = tau, so Var(A m) = E[(A m)^2] - tau^2
        # but m here is the conditional mean, so the design term is Var(A m(Z)).
        mean_score = float(np.sum(prob * score))
        design[i] = float(np.sum(prob * (score - mean_score) ** 2))
        noise[i] = float(np.sum(prob * (a ** 2) * vars_))
    return {
        "design_var": design,
        "utility_noise_var": noise,
        "total_var": design + noise,
    }


def release_means(
    main: np.ndarray | None = None,
    intercept: float = 0.0,
    sigma: float = 0.0,
) -> np.ndarray:
    """E[D | coalition] for all 32 coalitions, indexed by the client bitmask.

    D is the logistic release minus logistic(0). `main` and `intercept` replace
    the locked coefficients when a round's public state is not the frozen one.
    """
    from .game import GAMMA_01, MAIN

    coalitions = all_coalitions(N_CLIENTS)
    if main is None:
        main = MAIN
    main = np.asarray(main, dtype=np.float64)
    latent = coalitions @ main
    latent = latent + GAMMA_01 * coalitions[:, 0] * coalitions[:, 1] + float(intercept)
    return expected_utility(latent, float(sigma)) - U0


def expected_release(values: np.ndarray, probs: np.ndarray) -> float:
    """Expectation of a 32-vector under independent Bernoulli inclusions."""
    values = np.asarray(values, dtype=np.float64)
    probs = np.asarray(probs, dtype=np.float64)
    total = 0.0
    n_clients = int(probs.shape[0])
    for mask in range(1 << n_clients):
        weight = 1.0
        for j in range(n_clients):
            bit = (mask >> j) & 1
            weight *= float(probs[j]) if bit else (1.0 - float(probs[j]))
        total += weight * float(values[mask])
    return float(total)


def effects_from_product(values: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Add-one contrasts under independent inclusions with probabilities `probs`.

    Client i's coordinate is E[m | Z_i=1] - E[m | Z_i=0], with the other
    inclusions still following `probs`. A structural zero probability does not
    by itself set that client's target to zero: the coordinate remains the
    counterfactual add-one contrast. Callers that remove a client from the
    lottery must overwrite that coordinate.
    """
    values = np.asarray(values, dtype=np.float64)
    probs = np.asarray(probs, dtype=np.float64)
    n_clients = int(probs.shape[0])
    tau = np.zeros(n_clients, dtype=np.float64)
    for i in range(n_clients):
        acc = 0.0
        others = [j for j in range(n_clients) if j != i]
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
            acc += weight * (float(values[idx1]) - float(values[idx0]))
        tau[i] = acc
    return tau


def itt_and_received(
    p: float,
    dropout_rates: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Invitation ITT and received-participation contrast under one-sided dropout.

    Invitation Z_j is Bernoulli(p). Received participation is R_j = Z_j C_j with
    C_j Bernoulli(1 - r_j), independent of Z and of the coalition. The release
    depends on R only. Then the received contrast equals the ITT divided by the
    compliance probability 1 - r_j.
    """
    p = float(p)
    rates = np.asarray(dropout_rates, dtype=np.float64)
    if np.any(rates < 0.0) or np.any(rates >= 1.0):
        raise ValueError("dropout rates must lie in [0, 1)")
    q = p * (1.0 - rates)
    received = effects_from_product(values, q)
    itt = np.zeros(rates.shape[0], dtype=np.float64)
    for i in range(rates.shape[0]):
        probs_in = q.copy()
        probs_out = q.copy()
        probs_in[i] = 1.0 - float(rates[i])
        probs_out[i] = 0.0
        itt[i] = expected_release(values, probs_in) - expected_release(values, probs_out)
    return itt, received


def naive_receipt_population_limit(
    p: float,
    dropout_rates: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Population limit of IPW that treats receipt as Bernoulli(p).

    The weight is (R_i - p) / (p (1-p)) and the baseline is the mean release.
    Receipt is Bernoulli(p (1-r_i)), so the weight is not mean zero when r_i > 0.
    """
    p = float(p)
    rates = np.asarray(dropout_rates, dtype=np.float64)
    q = p * (1.0 - rates)
    mu = expected_release(values, q)
    denom = p * (1.0 - p)
    acc = np.zeros(rates.shape[0], dtype=np.float64)
    n_clients = int(rates.shape[0])
    for mask in range(1 << n_clients):
        weight = 1.0
        bits = np.empty(n_clients, dtype=np.float64)
        for j in range(n_clients):
            bit = float((mask >> j) & 1)
            bits[j] = bit
            weight *= float(q[j]) if bit == 1.0 else (1.0 - float(q[j]))
        centered = float(values[mask]) - mu
        for i in range(n_clients):
            a_weight = (bits[i] - p) / denom
            acc[i] += weight * a_weight * centered
    return acc


def replacement_effects(k: int, sigma: float = 0.0) -> np.ndarray:
    """Fixed-size-k contrast E[D | Z_i=1] - E[D | Z_i=0].

    Given Z_i=1 the other included set has size k-1. Given Z_i=0 it has size k.
    That shift is a replacement effect, not the Bernoulli add-one estimand.
    """
    from itertools import combinations

    if not (1 <= k < N_CLIENTS):
        raise ValueError("k out of range")
    coalitions = all_coalitions(N_CLIENTS)
    # Fixed-size oracle is for the nonprivate deterministic game unless sigma>0,
    # in which case m(z) already averages the Gaussian noise. No quorum.
    means = coalition_mean_delta(coalitions, sigma=sigma, quorum=None)
    tau = np.zeros(N_CLIENTS, dtype=np.float64)
    for i in range(N_CLIENTS):
        others = [j for j in range(N_CLIENTS) if j != i]
        m1_num = 0.0
        m1_den = 0
        if k - 1 >= 0:
            for combo in combinations(others, k - 1):
                z = np.zeros(N_CLIENTS, dtype=np.float64)
                z[i] = 1.0
                z[list(combo)] = 1.0
                idx = int(np.round(z) @ (1 << np.arange(N_CLIENTS)))
                m1_num += means[idx]
                m1_den += 1
        m0_num = 0.0
        m0_den = 0
        for combo in combinations(others, k):
            z = np.zeros(N_CLIENTS, dtype=np.float64)
            z[list(combo)] = 1.0
            idx = int(np.round(z) @ (1 << np.arange(N_CLIENTS)))
            m0_num += means[idx]
            m0_den += 1
        if m1_den == 0 or m0_den == 0:
            raise RuntimeError(f"empty fixed-size support for k={k}")
        tau[i] = m1_num / m1_den - m0_num / m0_den
    return tau
