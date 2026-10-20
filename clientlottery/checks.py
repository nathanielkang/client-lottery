"""Identity checks. These are not experimental results."""

from __future__ import annotations

import numpy as np

from .estimators import expanding_baseline, freedman_halfwidth, round_fixed_effect_diagnostic
from .game import N_CLIENTS, all_coalitions, coalition_mean_delta, draw_round, logistic
from .oracle import (
    conditional_mean_contrast,
    contrast_from_means,
    exact_psi_variance_b0,
    participation_effects,
    partner_conditional_effect,
    replacement_effects,
)


def _linear_oracle_matches_coefficients() -> None:
    """For D = c·Z with no noise, the Bernoulli add-one effect is c, at every p."""
    c = np.array([0.20, -0.10, 0.05, 0.00, 0.30], dtype=np.float64)
    coalitions = all_coalitions(5)
    means = coalitions @ c
    for p in (0.5, 0.3, 0.2):
        got = contrast_from_means(coalitions, means, p)
        if not np.allclose(got, c, atol=1e-12):
            raise AssertionError(f"linear oracle failed at p={p}: {got} vs {c}")


def _two_enumerations_agree() -> None:
    for p in (0.5, 0.3):
        for sigma, quorum in ((0.0, None), (1.0, None), (0.0, 3)):
            a = participation_effects(p, sigma=sigma, quorum=quorum)
            b = conditional_mean_contrast(p, sigma=sigma, quorum=quorum)
            if not np.allclose(a, b, atol=1e-10):
                raise AssertionError(
                    f"oracle mismatch p={p} sigma={sigma} quorum={quorum}: {a} vs {b}"
                )


def _interaction_identity() -> None:
    """D = Z_0 Z_1 has tau_0 = tau_1 = p and the other effects zero."""
    p = 0.3
    coalitions = ((np.arange(32)[:, None] >> np.arange(5)) & 1).astype(np.float64)
    means = coalitions[:, 0] * coalitions[:, 1]
    ones = coalitions.sum(axis=1)
    prob = (p ** ones) * ((1.0 - p) ** (5 - ones))
    tau = np.zeros(5)
    for i in range(5):
        inn = coalitions[:, i] == 1.0
        tau[i] = np.sum(prob[inn] * means[inn]) / p - np.sum(prob[~inn] * means[~inn]) / (1.0 - p)
    if abs(tau[0] - p) > 1e-12 or abs(tau[1] - p) > 1e-12:
        raise AssertionError(f"interaction identity failed: {tau}")
    if np.max(np.abs(tau[2:])) > 1e-12:
        raise AssertionError(f"interaction spilled: {tau}")


def _design_properties() -> None:
    tau_half = participation_effects(0.5, sigma=0.0, quorum=None)
    tau_policy = participation_effects(0.3, sigma=0.0, quorum=None)
    if np.allclose(tau_half, tau_policy, atol=1e-6):
        raise AssertionError("p=0.5 and p=0.3 oracles coincide; policy weight is invisible")
    if tau_half[2] >= -0.05:
        raise AssertionError(f"harmful client is not negative: {tau_half[2]}")
    if tau_half[0] <= 0.05:
        raise AssertionError(f"helpful client is not positive: {tau_half[0]}")
    m_in = partner_conditional_effect(1, 0, 1, p=0.5, sigma=0.0)
    m_out = partner_conditional_effect(1, 0, 0, p=0.5, sigma=0.0)
    if m_in < m_out + 0.05:
        raise AssertionError(f"complementarity too weak: in={m_in} out={m_out}")
    tau_dp = participation_effects(0.5, sigma=1.0, quorum=None)
    if np.allclose(tau_dp, tau_half, atol=1e-4):
        raise AssertionError("DP game matches the nonprivate game; Jensen gap missing")
    # Noise is not mean-zero on the utility scale: E[u(w+eta)] != u(w).
    raw = float(logistic(1.5))
    from .game import expected_utility

    smoothed = float(expected_utility(np.array(1.5), sigma=1.0))
    if abs(smoothed - raw) < 1e-4:
        raise AssertionError("logistic expectation ignored the noise")
    tau_q = participation_effects(0.5, sigma=0.0, quorum=3)
    if np.allclose(tau_q, tau_half, atol=1e-4):
        raise AssertionError("quorum no-op did not change the participation effect")
    repl = replacement_effects(2, sigma=0.0)
    if np.allclose(repl, tau_half, atol=1e-4):
        raise AssertionError("fixed-k replacement matches the add-one oracle")
    tau_marginal_rate = participation_effects(0.4, sigma=0.0, quorum=None)
    if np.allclose(repl, tau_marginal_rate, atol=1e-4):
        raise AssertionError("fixed-k replacement matches Bernoulli(k/n) add-one")
    # Separated effects at both policies. A near-tie has no rank guarantee.
    for label, tau in (("p=1/2", tau_half), ("p=0.3", tau_policy)):
        ordered = np.sort(tau)
        if np.min(np.diff(ordered)) < 0.03:
            raise AssertionError(f"oracle effects are not separated at {label}: {tau}")
    if np.array_equal(np.argsort(-tau_half), np.argsort(-tau_policy)):
        raise AssertionError("policy-weighted ranking matches uniform Banzhaf ranking")


def _baseline_uses_past_only() -> None:
    d = np.array([0.2, -0.4, 0.1, 0.0], dtype=np.float64)
    b = expanding_baseline(d)
    if b[0] != 0.0 or b[1] != 0.2:
        raise AssertionError(b)
    if abs(b[2] - (0.2 - 0.4) / 2) > 1e-12:
        raise AssertionError(b)
    if abs(b[3] - (0.2 - 0.4 + 0.1) / 3) > 1e-12:
        raise AssertionError(b)


def _round_fe_fails() -> None:
    rng = np.random.default_rng(0)
    z = (rng.random((24, N_CLIENTS)) < 0.5).astype(np.float64)
    diag = round_fixed_effect_diagnostic(z)
    if diag["identified"]:
        raise AssertionError("round FE was identified")
    if diag["gram_max_abs"] > 1e-12 or diag["residual_max_abs"] > 1e-12:
        raise AssertionError(diag)
    if diag["design_rank"] != 24:
        raise AssertionError(diag)


def _freedman_scales() -> None:
    width = freedman_halfwidth(4000, kappa=0.5, delta=0.05)
    # Forgetting the /T division would leave a radius of thousands.
    if not (0.2 < width < 0.8):
        raise AssertionError(f"freedman halfwidth out of the expected loose range: {width}")


def _variance_split_matches_monte_carlo() -> None:
    rng = np.random.default_rng(7)
    t_rounds = 40000
    for p, sigma, quorum in ((0.5, 0.0, None), (0.5, 1.0, None)):
        z, d, _ = draw_round(rng, t_rounds, p=p, sigma=sigma, quorum=quorum)
        a = (z - p) / (p * (1.0 - p))
        psi = a * d[:, None]
        exact = exact_psi_variance_b0(p, sigma=sigma, quorum=quorum)
        emp = np.var(psi, axis=0, ddof=0)
        if not np.allclose(emp, exact["total_var"], rtol=0.08, atol=0.02):
            raise AssertionError(
                f"variance split mismatch p={p} sigma={sigma}: emp={emp} exact={exact['total_var']}"
            )
        if sigma == 0.0 and np.max(exact["utility_noise_var"]) > 1e-12:
            raise AssertionError("nonprivate game reported utility-noise variance")
        if sigma > 0.0 and np.min(exact["utility_noise_var"]) <= 0.0:
            raise AssertionError("DP game reported no utility-noise variance")


def _ols_population_equals_participation_effect() -> None:
    """Independent Bernoulli OLS coefficient equals tau for any coalition utility."""
    coalitions = all_coalitions(5)
    cases = (
        (0.5, 0.0, None),
        (0.3, 0.0, None),
        (0.5, 1.0, None),
        (0.5, 0.0, 3),
    )
    for p, sigma, quorum in cases:
        means = coalition_mean_delta(coalitions, sigma=sigma, quorum=quorum)
        tau = contrast_from_means(coalitions, means, p)
        ones = coalitions.sum(axis=1)
        prob = (p ** ones) * ((1.0 - p) ** (5 - ones))
        x = np.column_stack([np.ones(coalitions.shape[0]), coalitions])
        gram = np.zeros((6, 6))
        rhs = np.zeros(6)
        for s in range(coalitions.shape[0]):
            gram += prob[s] * np.outer(x[s], x[s])
            rhs += prob[s] * x[s] * means[s]
        coef = np.linalg.solve(gram, rhs)[1:]
        if not np.allclose(coef, tau, atol=1e-10):
            raise AssertionError(f"OLS/tau mismatch p={p} sigma={sigma} quorum={quorum}: {coef} vs {tau}")


def _missing_battery_identities() -> None:
    """Dropout, reuse, Shapley, and drift-path identities. Not a results ledger."""
    from .missing_oracle import (
        add_one_from_empty,
        drift_strong_targets,
        dropout_population,
        eligibility_targets,
        leave_one_out,
        saturated_reuse,
        shapley_values,
    )
    from .missing_spec import DROPOUT_P, DROPOUT_RATES
    from .oracle import effects_from_product, itt_and_received, release_means

    values = release_means()
    direct = coalition_mean_delta(all_coalitions(5), sigma=0.0, quorum=None)
    if not np.allclose(values, direct, atol=1e-12):
        raise AssertionError("release_means drifted from the locked nonprivate game")
    ref_half = participation_effects(0.5, sigma=0.0, quorum=None)
    for p, ref in (
        (0.5, ref_half),
        (0.3, participation_effects(0.3, sigma=0.0, quorum=None)),
    ):
        got = effects_from_product(values, np.full(5, p))
        if not np.allclose(got, ref, atol=1e-10):
            raise AssertionError(f"product contrast mismatch at p={p}: {got} vs {ref}")

    zero_rates = np.zeros(5)
    itt0, received0 = itt_and_received(0.5, zero_rates, values)
    if not np.allclose(itt0, ref_half, atol=1e-10) or not np.allclose(received0, ref_half, atol=1e-10):
        raise AssertionError("full compliance did not recover the lottery effect")

    pop = dropout_population()
    if not np.allclose(pop["itt"], pop["received"] * pop["compliance"], atol=1e-10):
        raise AssertionError(
            f"ITT is not compliance times the received contrast: {pop['itt']} vs {pop['received']}"
        )
    if pop["itt"][0] <= 0.05 or pop["itt"][2] >= -0.03:
        raise AssertionError(f"dropout ITT signs/magnitudes are too weak: {pop['itt']}")
    naive_gap = np.abs(pop["naive_population_limit"] - pop["itt"])
    if float(np.mean(naive_gap)) <= 0.02 or float(naive_gap[2]) <= 0.05:
        raise AssertionError(f"naive receipt IPW is not biased enough: gap={naive_gap}")

    coalitions = all_coalitions(5)
    d = coalition_mean_delta(coalitions, sigma=0.0, quorum=None)
    tau_hat, se, n_obs = saturated_reuse(coalitions, d, 0.5)
    if n_obs != 32 or not np.allclose(tau_hat, ref_half, atol=1e-10):
        raise AssertionError(f"saturated reuse failed: n={n_obs} tau={tau_hat}")
    if np.max(se) > 1e-12:
        raise AssertionError(f"deterministic one-shot cells reported SE {se}")

    phi = shapley_values(values)
    if abs(float(np.sum(phi)) - float(values[(1 << 5) - 1] - values[0])) > 1e-10:
        raise AssertionError(f"Shapley efficiency failed: {phi}")
    loo = leave_one_out(values)
    add = add_one_from_empty(values)
    if np.allclose(phi, ref_half, atol=1e-6):
        raise AssertionError("Shapley matches the lottery effect; the games were not separated")
    if np.allclose(loo, ref_half, atol=1e-6) or np.allclose(add, ref_half, atol=1e-6):
        raise AssertionError("LOO or add-one matches the lottery effect")

    drift = drift_strong_targets(3000, p=0.5)
    path = drift["path_average"]
    last = drift["last_state"]
    if path[0] <= 0.05 or path[2] >= -0.03:
        raise AssertionError(f"drift_strong path-average signs failed: {path}")
    if abs(float(path[0] - last[0])) <= 0.03 or abs(float(path[2] - last[2])) <= 0.03:
        raise AssertionError(f"drift_strong does not move tau: path={path} last={last}")
    elig = eligibility_targets(3000, p=DROPOUT_P)
    if elig["path_average"][0] <= 0.05 or elig["path_average"][2] >= -0.03:
        raise AssertionError(f"eligibility path signs failed: {elig['path_average']}")
    if abs(float(elig["ineligible_block_tau"][4])) > 1e-15:
        raise AssertionError("ineligible client target was not zero")
    # rates argument is part of the design lock; unused math would hide a rename.
    if DROPOUT_RATES.shape != (5,):
        raise AssertionError("dropout rate vector length")
    if not (0.0 <= DROPOUT_P <= 1.0):
        raise AssertionError("dropout p")


def run_checks() -> dict:
    _linear_oracle_matches_coefficients()
    _interaction_identity()
    _two_enumerations_agree()
    _design_properties()
    _ols_population_equals_participation_effect()
    _baseline_uses_past_only()
    _round_fe_fails()
    _freedman_scales()
    _variance_split_matches_monte_carlo()
    _missing_battery_identities()
    return {"unit_checks": "pass"}
