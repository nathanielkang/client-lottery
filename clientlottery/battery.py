"""Missing-experiment battery. Does not write the T=50000 frozen ledger."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .checks import run_checks
from .estimators import (
    freedman_halfwidth,
    ipw_estimates,
    mean_abs_error,
    sign_of,
    spearman,
)
from .game import (
    CLIENT_ROLES,
    DP_SIGMA,
    GAME_ID,
    GAMMA_01,
    MAIN,
    N_CLIENTS,
    U0,
    draw_round,
    kappa_of,
    latent_of,
    logistic,
)
from .ledger import write_csv, write_dictionary, write_json
from .missing_oracle import (
    add_one_from_empty,
    drift_strong_targets,
    dropout_population,
    eligibility_targets,
    leave_one_out,
    phase_bounds,
    saturated_reuse,
    shapley_values,
)
from .missing_spec import (
    DROPOUT_P,
    DROPOUT_RATES,
    ELIGIBILITY_CLIENT,
    N_SIMS,
    PAPER_MIN_T,
    PAPER_SEEDS,
    PHASE_INTERCEPTS,
    PHASE_SCALES,
    RARE_P,
    RARE_T_CAP,
    RIDGE_ALPHA,
    SMOKE_SEEDS,
    SMOKE_T,
    SURROGATE_LABEL,
)
from .oracle import participation_effects

PREDECLARED_DELTA = 0.05
NORMAL_Z = 1.96


def _protocol_budget(protocol: str) -> tuple[int | None, tuple[int, ...], bool]:
    if protocol == "smoke_missing":
        return SMOKE_T, SMOKE_SEEDS, False
    if protocol == "paper_missing_battery":
        return None, PAPER_SEEDS, True
    raise ValueError(f"unknown protocol {protocol}")


def _rng(seed: int, code: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(code), 5602]))


def _role_rows(rows: list[dict]) -> dict[int, dict[str, dict]]:
    grouped: dict[int, dict[str, dict]] = {}
    for row in rows:
        grouped.setdefault(int(row["seed"]), {})[row["client_role"]] = row
    return grouped


def _mae_pair(estimate: np.ndarray, oracle: np.ndarray) -> float:
    return mean_abs_error(estimate, oracle)


def _sign_row(estimate: float, oracle: float) -> tuple[int | None, float | None, int | None]:
    if not np.isfinite(estimate) or not np.isfinite(oracle):
        return None, None, None
    return sign_of(estimate), abs(float(estimate) - float(oracle)), int(sign_of(estimate) == sign_of(oracle))


def _client_stub(i: int) -> dict:
    return {"client_id": i, "client_role": CLIENT_ROLES[i]}


def _empirical_se(psi: np.ndarray) -> np.ndarray:
    psi = np.asarray(psi, dtype=np.float64)
    if psi.shape[0] < 2:
        return np.full(psi.shape[1], np.nan)
    return np.std(psi, axis=0, ddof=1) / np.sqrt(psi.shape[0])


def _ridge_readout(z: np.ndarray, d: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Ridge of D on current participation, and on current plus lagged participation.

    The lagged coefficient is not a client score. The readout is the
    contemporaneous coefficient. At a frozen state the lag does not carry
    the pre-round state.
    """
    z = np.asarray(z, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    t_rounds, n_clients = z.shape
    lag = np.vstack([np.zeros((1, n_clients), dtype=np.float64), z[:-1]])
    current = _ridge_slopes(np.column_stack([np.ones(t_rounds), z]), d, alpha)
    both = _ridge_slopes(np.column_stack([np.ones(t_rounds), z, lag]), d, alpha)
    return current, both[:n_clients]


def _ridge_slopes(x: np.ndarray, d: np.ndarray, alpha: float) -> np.ndarray:
    penalty = np.eye(x.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    gram = x.T @ x + float(alpha) * penalty
    coef = np.linalg.solve(gram, x.T @ d)
    return coef[1:]


def _draw_dropout(rng: np.random.Generator, t_rounds: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    invited = (rng.random((t_rounds, N_CLIENTS)) < DROPOUT_P).astype(np.float64)
    comply = (rng.random((t_rounds, N_CLIENTS)) >= DROPOUT_RATES).astype(np.float64)
    received = invited * comply
    release = logistic(latent_of(received)) - U0
    return invited, received, release


def _draw_drift(rng: np.random.Generator, t_rounds: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    b1, b2 = phase_bounds(t_rounds)
    invited = (rng.random((t_rounds, N_CLIENTS)) < 0.5).astype(np.float64)
    release = np.empty(t_rounds, dtype=np.float64)
    slices = (slice(0, b1), slice(b1, b2), slice(b2, t_rounds))
    for sl, scale, intercept in zip(slices, PHASE_SCALES, PHASE_INTERCEPTS):
        coalition = invited[sl]
        main = float(scale) * MAIN
        latent = coalition @ main
        latent = latent + GAMMA_01 * coalition[:, 0] * coalition[:, 1] + float(intercept)
        release[sl] = logistic(latent) - U0
    return invited, release, b1, b2


def _draw_eligibility(rng: np.random.Generator, t_rounds: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    b1, b2 = phase_bounds(t_rounds)
    invited = (rng.random((t_rounds, N_CLIENTS)) < 0.5).astype(np.float64)
    invited[b1:b2, ELIGIBILITY_CLIENT] = 0.0
    release = logistic(latent_of(invited)) - U0
    return invited, release, b1, b2


def _zero_ineligible(psi: np.ndarray, b1: int, b2: int) -> np.ndarray:
    out = np.array(psi, copy=True)
    out[b1:b2, ELIGIBILITY_CLIENT] = 0.0
    return out


def _rare_length(oracle: np.ndarray) -> int | None:
    target = 0.80 * float(min(abs(oracle[0]), abs(oracle[2])))
    kappa = kappa_of(RARE_P)
    if target <= 0.0:
        return None
    if freedman_halfwidth(RARE_T_CAP, kappa, PREDECLARED_DELTA) > target:
        return None
    lo, hi = 1000, RARE_T_CAP
    while lo < hi:
        mid = (lo + hi) // 2
        if freedman_halfwidth(mid, kappa, PREDECLARED_DELTA) <= target:
            hi = mid
        else:
            lo = mid + 1
    return int(np.ceil(lo / 1000.0) * 1000)


def _dropout_rows(protocol: str, paper_flag: bool, seeds: tuple[int, ...], t_rounds: int) -> list[dict]:
    population = dropout_population()
    rows: list[dict] = []
    for seed in seeds:
        invited, received, release = _draw_dropout(_rng(seed, 11), t_rounds)
        itt_hat, _ = ipw_estimates(invited, release, DROPOUT_P, "expanding_mean_past_release")
        naive_hat, _ = ipw_estimates(received, release, DROPOUT_P, "expanding_mean_past_release")
        adjusted = itt_hat / population["compliance"]
        mae_itt = _mae_pair(itt_hat, population["itt"])
        mae_naive = _mae_pair(naive_hat, population["itt"])
        for i in range(N_CLIENTS):
            itt_sign, itt_err, itt_match = _sign_row(float(itt_hat[i]), float(population["itt"][i]))
            naive_sign, naive_err, naive_match = _sign_row(float(naive_hat[i]), float(population["itt"][i]))
            adj_sign, adj_err, adj_match = _sign_row(float(adjusted[i]), float(population["received"][i]))
            row = {
                "protocol": protocol,
                "paper_protocol": int(paper_flag),
                "seed": int(seed),
                "T": int(t_rounds),
                "setting": "frozen_dropout",
                "game_id": GAME_ID,
                "p_invitation": DROPOUT_P,
                "client_id": i,
                "client_role": CLIENT_ROLES[i],
                "dropout_rate": float(DROPOUT_RATES[i]),
                "compliance_rate": float(population["compliance"][i]),
                "estimand_itt": "invitation_effect_under_one_sided_dropout",
                "estimand_received_participation": "received_participation_effect_under_compliance_model",
                "oracle_itt": float(population["itt"][i]),
                "oracle_received": float(population["received"][i]),
                "ipw_itt": float(itt_hat[i]),
                "abs_error_ipw_itt": itt_err,
                "sign_match_ipw_itt": itt_match,
                "estimate_sign_ipw_itt": itt_sign,
                "naive_ipw_on_receipt": float(naive_hat[i]),
                "abs_error_naive_vs_itt": naive_err,
                "sign_match_naive_vs_itt": naive_match,
                "estimate_sign_naive": naive_sign,
                "naive_population_limit": float(population["naive_population_limit"][i]),
                "compliance_adjusted": float(adjusted[i]),
                "abs_error_compliance_vs_received": adj_err,
                "sign_match_compliance_vs_received": adj_match,
                "setting_mae_ipw_itt": mae_itt,
                "setting_mae_naive_vs_itt": mae_naive,
                "mae_gap_naive_minus_itt": mae_naive - mae_itt,
                "naive_uses_invitation_p_on_receipt": 1,
                "receipt_is_not_the_invitation": 1,
            }
            rows.append(row)
        print(f"SETTING_DONE setting=frozen_dropout seed={seed} T={t_rounds}", flush=True)
    return rows


def _reuse_block(
    protocol: str,
    paper_flag: bool,
    seed: int,
    t_rounds: int,
    setting: str,
    p: float,
    sigma: float,
    code: int,
    invited: np.ndarray | None = None,
    release: np.ndarray | None = None,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    if invited is None or release is None:
        invited, release, _ = draw_round(_rng(seed, code), t_rounds, p, sigma, None)
    oracle = participation_effects(p, sigma=sigma, quorum=None)
    ipw, psi = ipw_estimates(invited, release, p, "expanding_mean_past_release")
    reuse, reuse_se, n_obs = saturated_reuse(invited, release, p)
    if paper_flag and n_obs < (1 << N_CLIENTS):
        raise RuntimeError(f"{setting} p={p} seed={seed} saw {n_obs} coalitions")
    ipw_se = _empirical_se(psi)
    mae_ipw = _mae_pair(ipw, oracle)
    mae_reuse = _mae_pair(reuse, oracle)
    rows = []
    for i in range(N_CLIENTS):
        ipw_sign, ipw_err, ipw_match = _sign_row(float(ipw[i]), float(oracle[i]))
        reuse_sign, reuse_err, reuse_match = _sign_row(float(reuse[i]), float(oracle[i]))
        rows.append(
            {
                "protocol": protocol,
                "paper_protocol": int(paper_flag),
                "seed": int(seed),
                "T": int(t_rounds),
                "setting": setting,
                "p": float(p),
                "dp_sigma": float(sigma),
                "client_id": i,
                "client_role": CLIENT_ROLES[i],
                "estimand_lottery_participation_effect": float(oracle[i]),
                "single_client_ipw": float(ipw[i]),
                "multilinear_reuse": None if not np.isfinite(reuse[i]) else float(reuse[i]),
                "abs_error_ipw": ipw_err,
                "abs_error_reuse": reuse_err,
                "sign_match_ipw": ipw_match,
                "sign_match_reuse": reuse_match,
                "estimate_sign_ipw": ipw_sign,
                "estimate_sign_reuse": reuse_sign,
                "empirical_se_ipw": float(ipw_se[i]),
                "empirical_se_reuse": None if not np.isfinite(reuse_se[i]) else float(reuse_se[i]),
                "empirical_width_ipw": float(NORMAL_Z * ipw_se[i]),
                "empirical_width_reuse": None if not np.isfinite(reuse_se[i]) else float(NORMAL_Z * reuse_se[i]),
                "width_note": "normal_approximation_not_a_freedman_certificate",
                "setting_mae_ipw": mae_ipw,
                "setting_mae_reuse": mae_reuse,
                "n_distinct_coalitions_observed": n_obs,
                "extra_coalition_count": 0,
                "same_transcript": 1,
                "support_complete": int(n_obs == (1 << N_CLIENTS)),
            }
        )
    print(
        f"SETTING_DONE setting=reuse:{setting} p={p} sigma={sigma} seed={seed} "
        f"coalitions={n_obs} mae_ipw={mae_ipw:.6g} mae_reuse={mae_reuse:.6g}",
        flush=True,
    )
    return rows, ipw, oracle


def _shapley_rows(
    protocol: str,
    paper_flag: bool,
    seed: int,
    t_rounds: int,
    ipw: np.ndarray,
    lottery: np.ndarray,
    shapley: np.ndarray,
    loo: np.ndarray,
    addone: np.ndarray,
    rho: float,
) -> list[dict]:
    mae = _mae_pair(ipw, lottery)
    rows = []
    for i in range(N_CLIENTS):
        _, err, _ = _sign_row(float(ipw[i]), float(lottery[i]))
        rows.append(
            {
                "protocol": protocol,
                "paper_protocol": int(paper_flag),
                "seed": int(seed),
                "T": int(t_rounds),
                "setting": "frozen_nonprivate",
                "p": 0.5,
                "client_id": i,
                "client_role": CLIENT_ROLES[i],
                "estimand_lottery_participation_effect": float(lottery[i]),
                "ipw_on_transcript": float(ipw[i]),
                "abs_error_ipw_vs_lottery_oracle": err,
                "setting_mae_ipw_vs_lottery_oracle": mae,
                "estimand_shapley_or_loo": float(shapley[i]),
                "loo_full_minus_without": float(loo[i]),
                "addone_singleton_minus_empty": float(addone[i]),
                "spearman_lottery_tau_vs_shapley": rho,
                "extra_coalition_count": 32,
                "loo_coalition_count": 6,
                "addone_coalition_count": 6,
                "outside_private_transcript": 1,
                "would_cost_privacy_if_released": 1,
                "ipw_uses_training_transcript_only": 1,
                "discrepancy_vs_shapley_is_not_ipw_error": 1,
            }
        )
    return rows


def _surrogate_rows(
    protocol: str,
    paper_flag: bool,
    seed: int,
    t_rounds: int,
    transcript_ipw: np.ndarray,
    lottery: np.ndarray,
) -> list[dict]:
    rows = []
    for n_sim in N_SIMS:
        invited, release, _ = draw_round(_rng(seed, 4000 + int(n_sim)), int(n_sim), 0.5, 0.0, None)
        ridge, ridge_lag = _ridge_readout(invited, release, RIDGE_ALPHA)
        marginal, _ = ipw_estimates(invited, release, 0.5, "zero")
        rho_ridge = spearman(ridge, transcript_ipw)
        rho_marginal = spearman(marginal, transcript_ipw)
        for i in range(N_CLIENTS):
            rows.append(
                {
                    "protocol": protocol,
                    "paper_protocol": int(paper_flag),
                    "seed": int(seed),
                    "T_transcript": int(t_rounds),
                    "setting": "frozen_nonprivate",
                    "p": 0.5,
                    "client_id": i,
                    "client_role": CLIENT_ROLES[i],
                    "N_sim": int(n_sim),
                    "method_label": SURROGATE_LABEL,
                    "extra_simulations_not_training_transcript": 1,
                    "ridge_alpha": RIDGE_ALPHA,
                    "surrogate_ridge_coef": float(ridge[i]),
                    "surrogate_ridge_coef_with_lag": float(ridge_lag[i]),
                    "surrogate_extra_sim_marginal": float(marginal[i]),
                    "transcript_ipw": float(transcript_ipw[i]),
                    "estimand_lottery_participation_effect": float(lottery[i]),
                    "spearman_ridge_vs_transcript_ipw": rho_ridge,
                    "spearman_marginal_vs_transcript_ipw": rho_marginal,
                    "history_features": "lagged_participation_not_a_moving_state",
                }
            )
        print(f"SETTING_DONE setting=surrogate N_sim={n_sim} seed={seed}", flush=True)
    return rows


def _path_rows(
    protocol: str,
    paper_flag: bool,
    seed: int,
    t_rounds: int,
    setting: str,
    invited: np.ndarray,
    release: np.ndarray,
    b1: int,
    b2: int,
    targets: dict,
    zero_middle: bool,
) -> list[dict]:
    _, psi = ipw_estimates(invited, release, 0.5, "expanding_mean_past_release")
    if zero_middle:
        psi = _zero_ineligible(psi, b1, b2)
    full = psi.mean(axis=0)
    trailing = psi[b2:].mean(axis=0)
    path = np.asarray(targets["path_average"], dtype=np.float64)
    last = np.asarray(targets["last_state"], dtype=np.float64)
    phases = targets.get("phase_tau")
    if phases is None:
        phases = [
            targets["eligible_tau"],
            targets["ineligible_block_tau"],
            targets["eligible_tau"],
        ]
    mae = _mae_pair(full, path)
    hits = int(np.sum(np.sign(full) == np.sign(path)))
    one_step = targets.get("max_one_step_L")
    if one_step is None:
        one_step = np.maximum(np.abs(phases[1] - phases[0]), np.abs(phases[2] - phases[1]))
    dev = targets.get("mean_abs_dev_from_last")
    if dev is None:
        weights = np.asarray(targets["weights"], dtype=np.float64)
        dev = float(weights[0]) * np.abs(phases[0] - last) + float(weights[1]) * np.abs(phases[1] - last)
    rows = []
    for i in range(N_CLIENTS):
        _, err_path, match_path = _sign_row(float(full[i]), float(path[i]))
        _, err_last, _ = _sign_row(float(full[i]), float(last[i]))
        _, err_trail, match_trail = _sign_row(float(trailing[i]), float(last[i]))
        bound = float(one_step[i]) * float(t_rounds - 1) / 2.0
        rows.append(
            {
                "protocol": protocol,
                "paper_protocol": int(paper_flag),
                "seed": int(seed),
                "T": int(t_rounds),
                "setting": setting,
                "p": 0.5,
                "client_id": i,
                "client_role": CLIENT_ROLES[i],
                "phase_scales": "|".join(f"{float(v):.2f}" for v in PHASE_SCALES) if setting == "drift_strong" else "1.00|1.00|1.00",
                "phase_intercepts": "|".join(f"{float(v):.2f}" for v in PHASE_INTERCEPTS) if setting == "drift_strong" else "0|0|0",
                "phase_end_1": b1,
                "phase_end_2": b2,
                "trailing_window_rounds": int(t_rounds - b2),
                "oracle_path_average": float(path[i]),
                "oracle_last_state": float(last[i]),
                "oracle_phase1": float(phases[0][i]),
                "oracle_phase2": float(phases[1][i]),
                "oracle_phase3": float(phases[2][i]),
                "ipw_full": float(full[i]),
                "ipw_trailing": float(trailing[i]),
                "abs_error_full_vs_path_average": err_path,
                "sign_match_full_vs_path_average": match_path,
                "abs_gap_full_vs_last_state": err_last,
                "abs_gap_trailing_vs_last_state": err_trail,
                "sign_match_trailing_vs_last_state": match_trail,
                "max_one_step_L": float(one_step[i]),
                "drift_bound_L_w_minus_1_over_2": bound,
                "mean_abs_dev_from_last": float(dev[i]),
                "bound_is_loose": int(bound > 1.0),
                "setting_mae_full_vs_path": mae,
                "setting_sign_hits_path": hits,
                "ineligible_client": "" if not zero_middle else int(ELIGIBILITY_CLIENT),
                "ineligible_phase": "" if not zero_middle else "middle",
            }
        )
    print(f"SETTING_DONE setting={setting} seed={seed} T={t_rounds} mae_path={mae:.6g}", flush=True)
    return rows


def _rare_rows(protocol: str, paper_flag: bool, seeds: tuple[int, ...], t_rounds: int, oracle: np.ndarray) -> list[dict]:
    halfwidth = freedman_halfwidth(t_rounds, kappa_of(RARE_P), PREDECLARED_DELTA)
    rows = []
    for seed in seeds:
        invited, release, _ = draw_round(_rng(seed, 61), t_rounds, RARE_P, 0.0, None)
        estimate, _ = ipw_estimates(invited, release, RARE_P, "expanding_mean_past_release")
        mae = _mae_pair(estimate, oracle)
        for i in range(N_CLIENTS):
            _, err, match = _sign_row(float(estimate[i]), float(oracle[i]))
            certified = int(abs(float(oracle[i])) > halfwidth and match == 1)
            rows.append(
                {
                    "protocol": protocol,
                    "paper_protocol": int(paper_flag),
                    "seed": int(seed),
                    "T": int(t_rounds),
                    "setting": "frozen_rare_p01",
                    "p": RARE_P,
                    "kappa": kappa_of(RARE_P),
                    "client_id": i,
                    "client_role": CLIENT_ROLES[i],
                    "estimand_lottery_participation_effect": float(oracle[i]),
                    "ipw_forecast": float(estimate[i]),
                    "abs_error": err,
                    "sign_match": match,
                    "conservative_freedman_halfwidth": halfwidth,
                    "oracle_exceeds_halfwidth": int(abs(float(oracle[i])) > halfwidth),
                    "certified_sign": certified,
                    "setting_mae": mae,
                    "halfwidth_formula": "same conservative radius as the deterministic bound",
                }
            )
        print(f"SETTING_DONE setting=frozen_rare_p01 seed={seed} T={t_rounds}", flush=True)
    return rows


DROPOUT_COLUMNS = [
    ("protocol", "smoke_missing or paper_missing_battery. Not paper_large and not the T=50000 grid."),
    ("paper_protocol", "1 for paper_missing_battery. 0 for smoke. Does not replace gcp_T50000_with_drift."),
    ("seed", "Seed stream."),
    ("T", "Invitation rounds."),
    ("setting", "frozen_dropout."),
    ("game_id", "Locked synthetic game."),
    ("p_invitation", "Bernoulli invitation probability."),
    ("client_id", "0 helpful, 1 complement, 2 harmful, 3 moderate, 4 weak negative."),
    ("client_role", "Role locked to client_id."),
    ("dropout_rate", "Probability an invitation is not received. Harmful is the largest."),
    ("compliance_rate", "One minus the dropout rate."),
    ("estimand_itt", "Invitation effect under one-sided dropout."),
    ("estimand_received_participation", "Effect of receipt under the compliance model. Not the ITT."),
    ("oracle_itt", "Enumerated invitation effect."),
    ("oracle_received", "Enumerated received-participation effect. Equals the ITT divided by compliance."),
    ("ipw_itt", "IPW on the invitation indicator."),
    ("abs_error_ipw_itt", "Absolute error of invitation IPW against the ITT oracle."),
    ("sign_match_ipw_itt", "1 if the invitation IPW sign matches the ITT oracle."),
    ("estimate_sign_ipw_itt", "Sign of the invitation IPW."),
    ("naive_ipw_on_receipt", "IPW that treats receipt as a Bernoulli draw at the invitation probability."),
    ("abs_error_naive_vs_itt", "Absolute error of that naive score against the ITT oracle."),
    ("sign_match_naive_vs_itt", "1 if the naive score sign matches the ITT oracle."),
    ("estimate_sign_naive", "Sign of the naive score."),
    ("naive_population_limit", "Population limit of the naive score with the mean release as the baseline."),
    ("compliance_adjusted", "Invitation IPW divided by the known compliance rate."),
    ("abs_error_compliance_vs_received", "Absolute error of the compliance-adjusted score against the received-participation oracle."),
    ("sign_match_compliance_vs_received", "1 if that score matches the received-participation sign."),
    ("setting_mae_ipw_itt", "Mean absolute ITT error over the five clients."),
    ("setting_mae_naive_vs_itt", "Mean absolute naive-on-receipt error against the ITT oracle."),
    ("mae_gap_naive_minus_itt", "Naive MAE minus ITT MAE. Positive when the naive score is worse."),
    ("naive_uses_invitation_p_on_receipt", "Always 1."),
    ("receipt_is_not_the_invitation", "Always 1."),
]

REUSE_COLUMNS = [
    ("protocol", "smoke_missing or paper_missing_battery."),
    ("paper_protocol", "1 for the battery paper budget."),
    ("seed", "Seed stream."),
    ("T", "Rounds in the shared transcript."),
    ("setting", "frozen_nonprivate or frozen_dp."),
    ("p", "Lottery probability."),
    ("dp_sigma", "Latent noise before the logistic. 0 is the nonprivate release."),
    ("client_id", "Client index."),
    ("client_role", "Role locked to client_id."),
    ("estimand_lottery_participation_effect", "Oracle participation effect of this lottery."),
    ("single_client_ipw", "IPW that uses only that client's invitation column and the shared release."),
    ("multilinear_reuse", "Saturated coalition-mean plug-in on the same (Z, D) transcript."),
    ("abs_error_ipw", "Absolute error of single-client IPW against the lottery oracle."),
    ("abs_error_reuse", "Absolute error of the multilinear plug-in against the same oracle."),
    ("sign_match_ipw", "1 if IPW matches the lottery sign."),
    ("sign_match_reuse", "1 if the plug-in matches the lottery sign."),
    ("estimate_sign_ipw", "Sign of IPW."),
    ("estimate_sign_reuse", "Sign of the plug-in."),
    ("empirical_se_ipw", "Sample standard deviation of the IPW score, divided by sqrt(T)."),
    ("empirical_se_reuse", "Delta-method standard error of the coalition-mean plug-in, conditional on the realized coalitions."),
    ("empirical_width_ipw", "1.96 times empirical_se_ipw. A normal approximation, not a Freedman certificate."),
    ("empirical_width_reuse", "1.96 times empirical_se_reuse. Same caveat."),
    ("width_note", "normal_approximation_not_a_freedman_certificate."),
    ("setting_mae_ipw", "Mean absolute IPW error."),
    ("setting_mae_reuse", "Mean absolute plug-in error."),
    ("n_distinct_coalitions_observed", "How many of the 32 coalitions appear in the transcript."),
    ("extra_coalition_count", "Always 0. The plug-in does not query coalitions outside the transcript."),
    ("same_transcript", "Always 1. IPW and the plug-in share the draw."),
    ("support_complete", "1 if all 32 coalitions were observed."),
]

SHAPLEY_COLUMNS = [
    ("protocol", "smoke_missing or paper_missing_battery."),
    ("paper_protocol", "1 for the battery paper budget."),
    ("seed", "Seed of the lottery transcript used for IPW. Shapley does not use that transcript."),
    ("T", "Length of the lottery transcript."),
    ("setting", "frozen_nonprivate."),
    ("p", "Lottery probability of the IPW column."),
    ("client_id", "Client index."),
    ("client_role", "Role locked to client_id."),
    ("estimand_lottery_participation_effect", "Oracle lottery participation effect."),
    ("ipw_on_transcript", "IPW on the training transcript."),
    ("abs_error_ipw_vs_lottery_oracle", "Absolute IPW error against the lottery oracle. Not against Shapley."),
    ("setting_mae_ipw_vs_lottery_oracle", "Mean absolute IPW error against the lottery oracle."),
    ("estimand_shapley_or_loo", "Exact Shapley value of the coalition release game."),
    ("loo_full_minus_without", "Leave-one-out: full coalition minus the coalition with this client removed."),
    ("addone_singleton_minus_empty", "Singleton coalition minus the empty coalition."),
    ("spearman_lottery_tau_vs_shapley", "Rank correlation of the lottery oracle with Shapley. Not an IPW error."),
    ("extra_coalition_count", "32 oracle coalition utilities. Not taken from the training transcript."),
    ("loo_coalition_count", "6 utilities if leave-one-out is computed alone."),
    ("addone_coalition_count", "6 utilities if the singleton contrasts are computed alone."),
    ("outside_private_transcript", "Always 1."),
    ("would_cost_privacy_if_released", "Always 1. Releasing those aggregates would be an extra query."),
    ("ipw_uses_training_transcript_only", "Always 1."),
    ("discrepancy_vs_shapley_is_not_ipw_error", "Always 1."),
]

SURROGATE_COLUMNS = [
    ("protocol", "smoke_missing or paper_missing_battery."),
    ("paper_protocol", "1 for the battery paper budget."),
    ("seed", "Seed."),
    ("T_transcript", "Length of the training transcript whose IPW ranking is the comparison."),
    ("setting", "frozen_nonprivate."),
    ("p", "Lottery probability."),
    ("client_id", "Client index."),
    ("client_role", "Role locked to client_id."),
    ("N_sim", "Extra simulated lotteries at the frozen state. Not the training transcript."),
    ("method_label", "surrogate_not_FedCCEA_paper."),
    ("extra_simulations_not_training_transcript", "Always 1."),
    ("ridge_alpha", "Ridge penalty on the slopes. The intercept is not penalized."),
    ("surrogate_ridge_coef", "Coefficient on current participation in a ridge fit of the extra simulations."),
    ("surrogate_ridge_coef_with_lag", "Contemporaneous coefficient when lagged participation is also in the fit."),
    ("surrogate_extra_sim_marginal", "IPW marginal on the extra simulations."),
    ("transcript_ipw", "IPW on the separate training transcript."),
    ("estimand_lottery_participation_effect", "Lottery oracle. The surrogate is not scored as IPW error."),
    ("spearman_ridge_vs_transcript_ipw", "Rank correlation of the ridge readout with transcript IPW."),
    ("spearman_marginal_vs_transcript_ipw", "Rank correlation of the extra-simulation marginals with transcript IPW."),
    ("history_features", "Lagged participation. The state is frozen, so the lag is not a moving state."),
]

PATH_COLUMNS = [
    ("protocol", "smoke_missing or paper_missing_battery."),
    ("paper_protocol", "1 for the battery paper budget."),
    ("seed", "Seed."),
    ("T", "Rounds."),
    ("setting", "drift_strong or eligibility_block."),
    ("p", "Invitation probability outside an ineligible block."),
    ("client_id", "Client index."),
    ("client_role", "Role locked to client_id."),
    ("phase_scales", "Multipliers on the locked main effects, three phases."),
    ("phase_intercepts", "Public latent intercepts, three phases."),
    ("phase_end_1", "First index of phase 2."),
    ("phase_end_2", "First index of phase 3. The trailing window starts here."),
    ("trailing_window_rounds", "Length of phase 3."),
    ("oracle_path_average", "Length-weighted average of the three phase effects."),
    ("oracle_last_state", "Effect in phase 3."),
    ("oracle_phase1", "Effect in phase 1."),
    ("oracle_phase2", "Effect in phase 2. Zero for a client removed from the lottery."),
    ("oracle_phase3", "Effect in phase 3."),
    ("ipw_full", "IPW averaged over all rounds."),
    ("ipw_trailing", "IPW averaged over phase 3 only. The baseline still uses earlier releases."),
    ("abs_error_full_vs_path_average", "Absolute error of the full-window IPW against the path average."),
    ("sign_match_full_vs_path_average", "1 if the full-window IPW matches the path-average sign."),
    ("abs_gap_full_vs_last_state", "Absolute gap between the full-window IPW and the last-state effect. A different target."),
    ("abs_gap_trailing_vs_last_state", "Absolute gap between the trailing IPW and the last-state effect."),
    ("sign_match_trailing_vs_last_state", "1 if the trailing IPW matches the last-state sign."),
    ("max_one_step_L", "Largest one-step jump of this client's effect. Zero inside a phase."),
    ("drift_bound_L_w_minus_1_over_2", "Development-note bound L*(T-1)/2. Loose when the path is piecewise constant."),
    ("mean_abs_dev_from_last", "Average absolute distance of the phase effects from the last state."),
    ("bound_is_loose", "1 if L*(T-1)/2 exceeds 1."),
    ("setting_mae_full_vs_path", "Mean absolute error of the full-window IPW against the path average."),
    ("setting_sign_hits_path", "How many clients matched the path-average sign."),
    ("ineligible_client", "Client forced out in the middle phase. Empty for drift_strong."),
    ("ineligible_phase", "middle, or empty."),
]

RARE_COLUMNS = [
    ("protocol", "paper_missing_battery. Smoke does not run this length."),
    ("paper_protocol", "1."),
    ("seed", "Seed."),
    ("T", "Rounds chosen so the conservative half-width sits under 0.80 times the smaller of the helpful and harmful oracle effects."),
    ("setting", "frozen_rare_p01."),
    ("p", "0.1."),
    ("kappa", "min(p, 1-p)."),
    ("client_id", "Client index."),
    ("client_role", "Role locked to client_id."),
    ("estimand_lottery_participation_effect", "Oracle at p=0.1."),
    ("ipw_forecast", "IPW with the expanding baseline."),
    ("abs_error", "Absolute error against the lottery oracle."),
    ("sign_match", "1 if the IPW sign matches the oracle."),
    ("conservative_freedman_halfwidth", "Development-note radius. Not a sample-variance certificate."),
    ("oracle_exceeds_halfwidth", "1 if the absolute oracle effect exceeds that radius."),
    ("certified_sign", "1 if the oracle clears the radius and the sign matches."),
    ("setting_mae", "Mean absolute error."),
    ("halfwidth_formula", "Pointer to the development-note radius."),
]


def _both_sign_seeds(rows: list[dict], match_key: str) -> tuple[int, int]:
    grouped = _role_rows(rows)
    hits = 0
    for pair in grouped.values():
        ok = True
        for role in ("helpful_main", "harmful_main"):
            row = pair.get(role)
            if row is None or int(row.get(match_key) or 0) != 1:
                ok = False
        hits += int(ok)
    return hits, len(grouped)


def _mae_gap_seeds(rows: list[dict]) -> int:
    grouped = _role_rows(rows)
    hits = 0
    for pair in grouped.values():
        row = pair.get("helpful_main")
        if row is None:
            continue
        if float(row["setting_mae_naive_vs_itt"]) > float(row["setting_mae_ipw_itt"]):
            hits += 1
    return hits


def _evaluate(
    dropout_rows: list[dict],
    reuse_rows: list[dict],
    shapley_rows: list[dict],
    surrogate_rows: list[dict],
    drift_rows: list[dict],
    t_rounds: int,
    paper_flag: bool,
) -> dict:
    sign_hits, n_seeds = _both_sign_seeds(dropout_rows, "sign_match_ipw_itt")
    gap_hits = _mae_gap_seeds(dropout_rows)
    # Same seeds: a seed counts only when both signs match and the MAE gap is positive.
    grouped = _role_rows(dropout_rows)
    same = 0
    for pair in grouped.values():
        helpful = pair.get("helpful_main")
        harmful = pair.get("harmful_main")
        if helpful is None or harmful is None:
            continue
        signs = int(helpful["sign_match_ipw_itt"]) == 1 and int(harmful["sign_match_ipw_itt"]) == 1
        gap = float(helpful["setting_mae_naive_vs_itt"]) > float(helpful["setting_mae_ipw_itt"])
        same += int(signs and gap)
    item_a = bool(paper_flag) and same >= 4 and n_seeds >= 5

    reuse_ok = bool(reuse_rows) and all(int(row["same_transcript"]) == 1 for row in reuse_rows)
    mae_ok = all(np.isfinite(float(row["setting_mae_ipw"])) and np.isfinite(float(row["setting_mae_reuse"])) for row in reuse_rows)
    length_ok = (not paper_flag) or all(int(row["T"]) >= PAPER_MIN_T for row in reuse_rows)
    item_b = reuse_ok and mae_ok and length_ok and any(row["setting"] == "frozen_nonprivate" and abs(float(row["p"]) - 0.5) < 1e-12 for row in reuse_rows)

    item_c = bool(shapley_rows) and all(
        int(row["extra_coalition_count"]) == 32
        and row["estimand_lottery_participation_effect"] != ""
        and np.isfinite(float(row["abs_error_ipw_vs_lottery_oracle"]))
        and np.isfinite(float(row["estimand_shapley_or_loo"]))
        for row in shapley_rows
    )

    sim_values = {int(row["N_sim"]) for row in surrogate_rows}
    item_d = sim_values >= {200, 1000} and all(row["method_label"] == SURROGATE_LABEL for row in surrogate_rows)

    drift_hits, drift_seeds = _both_sign_seeds(drift_rows, "sign_match_full_vs_path_average")
    item_e = bool(paper_flag) and drift_hits >= 4 and drift_seeds >= 5

    return {
        "A_dropout_itt_beats_naive": item_a,
        "A_sign_and_gap_seeds": same,
        "A_itt_sign_seeds": sign_hits,
        "A_mae_gap_seeds": gap_hits,
        "B_reuse_rows": item_b,
        "C_shapley_loo_rows": item_c,
        "D_surrogate_rows": item_d,
        "E_drift_strong_path_signs": item_e,
        "E_path_sign_seeds": drift_hits,
        "T": int(t_rounds),
        "paper_protocol": bool(paper_flag),
    }


def _column_file(path: Path, columns: list[tuple[str, str]], title: str) -> None:
    write_dictionary(path, columns, title)


def run_missing_battery(
    protocol: str,
    outdir: Path,
    execution_host: str = "local",
    t_rounds: int | None = None,
    seeds: tuple[int, ...] | None = None,
) -> dict:
    """Write the missing-experiment ledgers. Refuses the T=50000 directory."""
    outdir = Path(outdir)
    if "gcp_T50000" in outdir.resolve().as_posix():
        raise RuntimeError("refusing to write the missing battery over gcp_T50000_with_drift")
    default_t, default_seeds, paper_flag = _protocol_budget(protocol)
    if seeds is None:
        seeds = default_seeds
    else:
        seeds = tuple(int(seed) for seed in seeds)
    if protocol == "paper_missing_battery":
        if t_rounds is None or int(t_rounds) < PAPER_MIN_T:
            raise ValueError(f"paper_missing_battery requires T>={PAPER_MIN_T}")
        if seeds != tuple(PAPER_SEEDS):
            raise ValueError("paper_missing_battery uses seeds 0 1 2 3 4")
    if t_rounds is None:
        t_rounds = default_t
    t_rounds = int(t_rounds)
    check_status = run_checks()

    dropout_rows = _dropout_rows(protocol, paper_flag, seeds, t_rounds)
    reuse_rows: list[dict] = []
    shapley_rows: list[dict] = []
    surrogate_rows: list[dict] = []
    values = dropout_population()["values"]
    shapley = shapley_values(values)
    loo = leave_one_out(values)
    addone = add_one_from_empty(values)
    lottery_half = participation_effects(0.5, sigma=0.0, quorum=None)
    rho_shapley = spearman(lottery_half, shapley)

    for seed in seeds:
        block, ipw, oracle = _reuse_block(
            protocol, paper_flag, seed, t_rounds, "frozen_nonprivate", 0.5, 0.0, 21
        )
        reuse_rows.extend(block)
        shapley_rows.extend(
            _shapley_rows(protocol, paper_flag, seed, t_rounds, ipw, oracle, shapley, loo, addone, rho_shapley)
        )
        surrogate_rows.extend(_surrogate_rows(protocol, paper_flag, seed, t_rounds, ipw, oracle))
        for setting, p, sigma, code in (
            ("frozen_nonprivate", 0.3, 0.0, 22),
            ("frozen_dp", 0.5, DP_SIGMA, 23),
        ):
            block, _, _ = _reuse_block(protocol, paper_flag, seed, t_rounds, setting, p, sigma, code)
            reuse_rows.extend(block)

    drift_targets = drift_strong_targets(t_rounds, p=0.5)
    elig_targets = eligibility_targets(t_rounds, p=0.5)
    drift_rows: list[dict] = []
    elig_rows: list[dict] = []
    for seed in seeds:
        invited, release, b1, b2 = _draw_drift(_rng(seed, 51), t_rounds)
        drift_rows.extend(
            _path_rows(
                protocol, paper_flag, seed, t_rounds, "drift_strong",
                invited, release, b1, b2, drift_targets, zero_middle=False,
            )
        )
        invited, release, b1, b2 = _draw_eligibility(_rng(seed, 52), t_rounds)
        elig_rows.extend(
            _path_rows(
                protocol, paper_flag, seed, t_rounds, "eligibility_block",
                invited, release, b1, b2, elig_targets, zero_middle=True,
            )
        )

    rare_oracle = participation_effects(RARE_P, sigma=0.0, quorum=None)
    rare_T = _rare_length(rare_oracle) if paper_flag else None
    rare_rows: list[dict] = []
    if rare_T is not None:
        rare_rows = _rare_rows(protocol, paper_flag, seeds, rare_T, rare_oracle)

    checklist = _evaluate(dropout_rows, reuse_rows, shapley_rows, surrogate_rows, drift_rows, t_rounds, paper_flag)
    population = dropout_population()
    outdir.mkdir(parents=True, exist_ok=True)
    write_csv(outdir / "dropout_results.csv", DROPOUT_COLUMNS, dropout_rows)
    write_csv(outdir / "reuse_results.csv", REUSE_COLUMNS, reuse_rows)
    write_csv(outdir / "shapley_loo_results.csv", SHAPLEY_COLUMNS, shapley_rows)
    write_csv(outdir / "surrogate_results.csv", SURROGATE_COLUMNS, surrogate_rows)
    write_csv(outdir / "drift_strong_results.csv", PATH_COLUMNS, drift_rows)
    write_csv(outdir / "eligibility_block_results.csv", PATH_COLUMNS, elig_rows)
    if rare_rows:
        write_csv(outdir / "rare_p_results.csv", RARE_COLUMNS, rare_rows)
    _column_file(outdir / "DROPOUT_COLUMNS.md", DROPOUT_COLUMNS, "Dropout ledger")
    _column_file(outdir / "REUSE_COLUMNS.md", REUSE_COLUMNS, "Sample-reuse ledger")
    _column_file(outdir / "SHAPLEY_COLUMNS.md", SHAPLEY_COLUMNS, "Shapley and leave-one-out ledger")
    _column_file(outdir / "SURROGATE_COLUMNS.md", SURROGATE_COLUMNS, "Extra-simulation surrogate ledger")
    _column_file(outdir / "PATH_COLUMNS.md", PATH_COLUMNS, "Drift and eligibility ledger")
    if rare_rows:
        _column_file(outdir / "RARE_COLUMNS.md", RARE_COLUMNS, "Rare participation ledger")

    summary = {
        "protocol": protocol,
        "paper_protocol": paper_flag,
        "execution_host": execution_host,
        "game_id": GAME_ID,
        "T": int(t_rounds),
        "seeds": list(seeds),
        "unit_checks": check_status,
        "does_not_replace": "results/gcp_T50000_with_drift",
        "checklist": checklist,
        "dropout": {
            "setting": "frozen_dropout",
            "p_invitation": DROPOUT_P,
            "dropout_rates": DROPOUT_RATES,
            "compliance_model": (
                "R_j = Z_j C_j with C_j Bernoulli(1-r_j) independent of the invitation. "
                "The release depends on R only. The received contrast equals the ITT "
                "divided by (1-r_j). Naive IPW puts the invitation probability on R."
            ),
            "oracle_itt": population["itt"],
            "oracle_received": population["received"],
            "naive_population_limit": population["naive_population_limit"],
        },
        "reuse": {
            "extra_coalition_count": 0,
            "same_transcript": True,
            "empirical_width": "1.96 * SE, not a Freedman certificate",
        },
        "shapley": {
            "extra_coalition_count": 32,
            "spearman_lottery_tau_vs_shapley": rho_shapley,
            "shapley": shapley,
            "loo": loo,
            "addone": addone,
            "lottery_oracle_p_half": lottery_half,
        },
        "surrogate": {
            "label": SURROGATE_LABEL,
            "N_sim": list(N_SIMS),
            "ridge_alpha": RIDGE_ALPHA,
            "extra_simulations_not_training_transcript": True,
        },
        "drift_strong": {
            "phase_scales": PHASE_SCALES,
            "phase_intercepts": PHASE_INTERCEPTS,
            "path_average": drift_targets["path_average"],
            "last_state": drift_targets["last_state"],
            "bound_note": "L*(T-1)/2 is valid and loose for a piecewise-constant path. mean_abs_dev_from_last is the path term.",
        },
        "eligibility_block": {
            "client": ELIGIBILITY_CLIENT,
            "phase": "middle",
            "path_average": elig_targets["path_average"],
        },
        "rare": {
            "ran": bool(rare_rows),
            "p": RARE_P,
            "T": rare_T,
            "oracle": rare_oracle,
            "reason_if_skipped": None if rare_rows or not paper_flag else "half-width stays above the target inside the round cap",
            "skipped_on_smoke": not paper_flag,
        },
        "n_dropout_rows": len(dropout_rows),
        "n_reuse_rows": len(reuse_rows),
        "n_shapley_rows": len(shapley_rows),
        "n_surrogate_rows": len(surrogate_rows),
        "n_drift_rows": len(drift_rows),
        "n_eligibility_rows": len(elig_rows),
        "n_rare_rows": len(rare_rows),
    }
    summary = _jsonable(summary)
    write_json(outdir / "experiment_summary.json", summary)
    note = [
        "# Protocol ledger",
        "",
        f"- protocol: `{protocol}`",
        f"- T: {t_rounds}",
        f"- seeds: {list(seeds)}",
        f"- game_id: {GAME_ID}",
        "- This directory is not `gcp_T50000_with_drift`.",
        "- frozen_dropout: invitation Z is Bernoulli(p); receipt is Z times an independent compliance bit. Harmful clients drop more.",
        "- ITT IPW uses Z. Naive IPW uses receipt with the invitation probability. Compliance-adjusted IPW divides the ITT score by (1-r).",
        "- multilinear_reuse pools coalition means on the same transcript. extra_coalition_count is 0.",
        "- Shapley, leave-one-out, and add-one use 32 oracle coalition utilities outside the transcript. Their gap from IPW is not an IPW error.",
        f"- Surrogate label: `{SURROGATE_LABEL}`. N_sim in {list(N_SIMS)}. Those draws are not the training transcript.",
        "- drift_strong: three public phases of main-effect scale and intercept. Full-window IPW targets the path average. The trailing window is the last phase.",
        "- The bound L*(T-1)/2 is loose for a piecewise-constant path.",
        "- eligibility_block: client 4 is out of the lottery in the middle phase. That is not drift_strong.",
        f"- rare p={RARE_P}: T={rare_T}. Smoke does not run it." if paper_flag else "- rare participation is not part of smoke_missing.",
        "- Empirical widths are normal approximations. They are not Freedman certificates.",
        "",
    ]
    (outdir / "PROTOCOL_LEDGER.md").write_text("\n".join(note), encoding="utf-8")
    return summary


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value
