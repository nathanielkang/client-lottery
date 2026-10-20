"""Run the frozen-state grid and write ledgers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .checks import run_checks
from .estimators import (
    freedman_halfwidth,
    inclusion_exclusion,
    ipw_estimates,
    linear_regression,
    mean_abs_error,
    plugin_halfwidth_not_certified,
    ranks_descending,
    round_fixed_effect_diagnostic,
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
    PAPER_SEEDS,
    PAPER_T,
    SMOKE_SEEDS,
    SMOKE_T,
    DRIFT_AMPLITUDE,
    DRIFT_SETTING_NAME,
    bernoulli_settings,
    draw_fixed_size,
    draw_round,
    fixed_size_outcomes,
    slow_intercept,
)
from .ledger import (
    BERNOULLI_COLUMNS,
    FIXEDK_COLUMNS,
    write_csv,
    write_dictionary,
    write_json,
)
from .oracle import (
    exact_psi_variance_b0,
    participation_effects,
    path_average_participation_effects,
    replacement_effects,
)
from .winbar import HALFWIDTH_FORMULA, evaluate_win_bar

PREDECLARED_DELTA = 0.05
SETTING_CODE = {
    ("frozen_nonprivate", 0.5): 1,
    ("frozen_nonprivate", 0.3): 2,
    ("frozen_dp", 0.5): 3,
    ("frozen_dp", 0.3): 4,
    ("frozen_quorum_noop", 0.5): 5,
    ("drift_nonprivate", 0.5): 6,
}


def _protocol_budget(protocol: str) -> tuple[int | None, tuple[int, ...], bool]:
    if protocol == "smoke":
        return SMOKE_T, SMOKE_SEEDS, False
    if protocol == "paper":
        return PAPER_T, PAPER_SEEDS, True
    if protocol == "paper_large":
        # T is required from the caller so the ledger names the window.
        return None, PAPER_SEEDS, True
    raise ValueError(f"unknown protocol {protocol}")


def _rng(seed: int, code: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(code), 5601]))


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


def _blank_intervals() -> dict:
    return {
        "conservative_freedman_halfwidth": None,
        "covered_by_conservative_freedman": None,
        "bonferroni_freedman_halfwidth": None,
        "covered_by_bonferroni_freedman": None,
        "plugin_halfwidth_not_certified": None,
        "plugin_covered_not_certified": None,
    }


def _ipw_intervals(psi_i: np.ndarray, error: float, kappa: float, t_rounds: int) -> dict:
    marginal = freedman_halfwidth(t_rounds, kappa, PREDECLARED_DELTA)
    simultaneous = freedman_halfwidth(t_rounds, kappa, PREDECLARED_DELTA / N_CLIENTS)
    plugin = plugin_halfwidth_not_certified(psi_i, PREDECLARED_DELTA, kappa)
    return {
        "conservative_freedman_halfwidth": marginal,
        "covered_by_conservative_freedman": int(abs(error) <= marginal + 1e-12),
        "bonferroni_freedman_halfwidth": simultaneous,
        "covered_by_bonferroni_freedman": int(abs(error) <= simultaneous + 1e-12),
        "plugin_halfwidth_not_certified": plugin,
        "plugin_covered_not_certified": int(np.isfinite(plugin) and abs(error) <= plugin + 1e-12),
    }


def _base_row(protocol, paper_flag, seed, t_rounds, setting, oracle, nonprivate, gap) -> dict:
    return {
        "protocol": protocol,
        "paper_protocol": int(paper_flag),
        "seed": int(seed),
        "T": int(t_rounds),
        "window_w": int(t_rounds),
        "n_clients": N_CLIENTS,
        "frozen_state": 0 if float(setting.drift_amplitude) > 0.0 else 1,
        "kappa": setting.kappa,
        "p": setting.p,
        "policy_name": setting.policy,
        "setting": setting.name,
        "game_id": GAME_ID,
        "dp_sigma": setting.sigma,
        "quorum": "" if setting.quorum is None else int(setting.quorum),
        "quorum_label": setting.quorum_label,
        "oracle_tau_vector": oracle,
        "nonprivate_oracle_vector": nonprivate,
        "gap_vector": gap,
    }


def _client_common(i: int, oracle_i: float, nonprivate_i: float, gap_i: float, oracle_rank, inclusion, exclusion) -> dict:
    return {
        "client_id": i,
        "client_role": CLIENT_ROLES[i],
        "oracle_tau": float(oracle_i),
        "oracle_sign": sign_of(float(oracle_i)),
        "oracle_rank": oracle_rank[i],
        "nonprivate_oracle_tau": float(nonprivate_i),
        "gap_to_nonprivate_game": float(gap_i),
        "inclusion_count": int(inclusion[i]),
        "exclusion_count": int(exclusion[i]),
        "predeclared_delta": PREDECLARED_DELTA,
    }


def _finish_estimator_rows(
    skeleton: dict,
    name: str,
    note_targets: int,
    b_rule: str,
    estimate: np.ndarray,
    identified_client: np.ndarray,
    failure_reason: str,
    design_rank,
    design_ncols,
    gram_max,
    rank_check: str,
    binary_variation: np.ndarray,
    intervals_by_client: list[dict] | None,
    shared_tail: dict,
) -> list[dict]:
    oracle = skeleton["oracle_tau_vector"]
    est_rank = ranks_descending(estimate)
    mae = mean_abs_error(estimate, oracle)
    corr = spearman(estimate, oracle)
    hits = 0
    rows = []
    for i in range(N_CLIENTS):
        est_i = float(estimate[i]) if np.isfinite(estimate[i]) else None
        if est_i is None:
            sign_match = None
            abs_error = None
            est_sign = None
        else:
            est_sign = sign_of(est_i)
            sign_match = int(est_sign == sign_of(float(oracle[i])))
            abs_error = abs(est_i - float(oracle[i]))
            hits += sign_match
        intervals = intervals_by_client[i] if intervals_by_client else _blank_intervals()
        row = {
            "protocol": skeleton["protocol"],
            "paper_protocol": skeleton["paper_protocol"],
            "seed": skeleton["seed"],
            "T": skeleton["T"],
            "window_w": skeleton["window_w"],
            "n_clients": skeleton["n_clients"],
            "frozen_state": skeleton["frozen_state"],
            "kappa": skeleton["kappa"],
            "p": skeleton["p"],
            "policy_name": skeleton["policy_name"],
            "setting": skeleton["setting"],
            "game_id": skeleton["game_id"],
            "dp_sigma": skeleton["dp_sigma"],
            "quorum": skeleton["quorum"],
            "quorum_label": skeleton["quorum_label"],
            "estimator": name,
            "targets_participation_effect": note_targets,
            "client_id": shared_tail[i]["client_id"],
            "client_role": shared_tail[i]["client_role"],
            "oracle_tau": shared_tail[i]["oracle_tau"],
            "oracle_sign": shared_tail[i]["oracle_sign"],
            "oracle_rank": shared_tail[i]["oracle_rank"],
            "nonprivate_oracle_tau": shared_tail[i]["nonprivate_oracle_tau"],
            "gap_to_nonprivate_game": shared_tail[i]["gap_to_nonprivate_game"],
            "estimate": est_i,
            "estimate_sign": est_sign if est_i is not None else None,
            "estimate_rank": None if est_i is None else est_rank[i],
            "abs_error": abs_error,
            "sign_match": sign_match,
            "identified": int(bool(identified_client[i])),
            "realized_binary_variation": int(bool(binary_variation[i])),
            "failure_reason": "" if identified_client[i] else failure_reason,
            "design_rank": design_rank,
            "design_ncols": design_ncols,
            "gram_max_abs": gram_max,
            "rank_check": rank_check,
            "inclusion_count": shared_tail[i]["inclusion_count"],
            "exclusion_count": shared_tail[i]["exclusion_count"],
            "effective_overlap_setting": skeleton["effective_overlap_setting"],
            "n_aborted_included": skeleton["n_aborted_included"],
            "draws_dropped": 0,
            "max_abs_D": skeleton["max_abs_D"],
            "setting_mae": mae,
            "setting_spearman": corr,
            "setting_sign_hits": hits if np.isfinite(estimate).any() else None,
            "b_rule": b_rule,
            "predeclared_delta": shared_tail[i]["predeclared_delta"],
        }
        row.update(intervals)
        rows.append(row)
    # setting_sign_hits was counted inside the loop but written per row before the loop
    # finished. Fill the completed count.
    for row in rows:
        row["setting_sign_hits"] = hits if np.isfinite(estimate).any() else None
    return rows


def _one_bernoulli(protocol, paper_flag, seed, t_rounds, setting) -> tuple[list[dict], dict]:
    code = SETTING_CODE[(setting.name, float(setting.p))]
    rng = _rng(seed, code)
    intercepts = None
    if float(setting.drift_amplitude) > 0.0:
        intercepts = slow_intercept(t_rounds, setting.drift_amplitude)
        z, d, n_aborted = draw_round(
            rng, t_rounds, setting.p, setting.sigma, setting.quorum, intercept=intercepts
        )
    else:
        z, d, n_aborted = draw_round(rng, t_rounds, setting.p, setting.sigma, setting.quorum)
    if d.shape[0] != t_rounds or z.shape != (t_rounds, N_CLIENTS):
        raise RuntimeError("transcript changed length")
    if np.max(np.abs(d)) > 1.0 + 1e-9:
        raise RuntimeError(f"|D| exceeded 1: {np.max(np.abs(d))}")
    if intercepts is None:
        oracle = participation_effects(setting.p, sigma=setting.sigma, quorum=setting.quorum)
        nonprivate = participation_effects(setting.p, sigma=0.0, quorum=None)
    else:
        oracle = path_average_participation_effects(
            setting.p, setting.sigma, setting.quorum, intercepts
        )
        # Same nonprivate moving game. The gap column stays a privacy-game gap.
        nonprivate = oracle.copy()
    gap = oracle - nonprivate
    inclusion = z.sum(axis=0)
    exclusion = t_rounds - inclusion
    rates = np.minimum(inclusion, exclusion) / float(t_rounds)
    overlap = float(np.min(rates))
    skeleton = _base_row(protocol, paper_flag, seed, t_rounds, setting, oracle, nonprivate, gap)
    skeleton["effective_overlap_setting"] = overlap
    skeleton["n_aborted_included"] = n_aborted
    skeleton["max_abs_D"] = float(np.max(np.abs(d)))
    oracle_rank = ranks_descending(oracle)
    binary = (inclusion > 0) & (exclusion > 0)
    shared = [
        _client_common(i, oracle[i], nonprivate[i], gap[i], oracle_rank, inclusion, exclusion)
        for i in range(N_CLIENTS)
    ]
    rows: list[dict] = []

    forecast, psi_f = ipw_estimates(z, d, setting.p, "expanding_mean_past_release")
    zero, psi_0 = ipw_estimates(z, d, setting.p, "zero")
    if np.max(np.abs(expanding_check(d))) > 1.0 + 1e-9:
        raise RuntimeError("baseline left [-1, 1]")
    intervals_f = [
        _ipw_intervals(psi_f[:, i], abs(float(forecast[i] - oracle[i])), setting.kappa, t_rounds)
        for i in range(N_CLIENTS)
    ]
    intervals_0 = [
        _ipw_intervals(psi_0[:, i], abs(float(zero[i] - oracle[i])), setting.kappa, t_rounds)
        for i in range(N_CLIENTS)
    ]
    ones = np.ones(N_CLIENTS, dtype=bool)
    rows.extend(
        _finish_estimator_rows(
            skeleton, "ipw_forecast", 1, "expanding_mean_past_release", forecast, ones, "",
            "", "", None, "", binary, intervals_f, shared,
        )
    )
    rows.extend(
        _finish_estimator_rows(
            skeleton, "ipw_b0", 1, "zero", zero, ones, "",
            "", "", None, "", binary, intervals_0, shared,
        )
    )

    ie, ie_ok = inclusion_exclusion(z, d)
    rows.extend(
        _finish_estimator_rows(
            skeleton, "inclusion_exclusion", 1, "not_applicable", ie, ie_ok,
            "no_finite_sample_overlap", "", "", None, "", binary, None, shared,
        )
    )

    ols, ols_rank, ols_ncols, ols_ok = linear_regression(z, d)
    ols_ident = np.full(N_CLIENTS, ols_ok, dtype=bool)
    rows.extend(
        _finish_estimator_rows(
            skeleton, "linear_regression", 1, "not_applicable", ols, ols_ident,
            "participation_design_rank_deficient", ols_rank, ols_ncols, None, "",
            binary, None, shared,
        )
    )

    diag = round_fixed_effect_diagnostic(z)
    if diag["gram_max_abs"] > 1e-8 or diag["identified"]:
        raise RuntimeError(f"saturated round effects did not fail: {diag}")
    empty = np.full(N_CLIENTS, np.nan)
    none_ident = np.zeros(N_CLIENTS, dtype=bool)
    rows.extend(
        _finish_estimator_rows(
            skeleton, "round_fe_saturated", 0, "not_applicable", empty, none_ident,
            diag["failure_reason"], diag["design_rank"], diag["design_ncols"],
            diag["gram_max_abs"], diag["rank_check"], binary, None, shared,
        )
    )

    if intercepts is None:
        split = exact_psi_variance_b0(setting.p, sigma=setting.sigma, quorum=setting.quorum)
    else:
        split = {
            "not_the_frozen_split": True,
            "reason": "variance split in the note is for a frozen state; this setting moves the intercept",
        }
    summary = {
        "setting": setting.name,
        "p": setting.p,
        "policy_name": setting.policy,
        "dp_sigma": setting.sigma,
        "quorum": setting.quorum,
        "quorum_label": setting.quorum_label,
        "drift_amplitude": float(setting.drift_amplitude),
        "frozen_state": intercepts is None,
        "seed": int(seed),
        "T": int(t_rounds),
        "n_aborted_included": n_aborted,
        "draws_dropped": 0,
        "effective_overlap": overlap,
        "max_abs_D": skeleton["max_abs_D"],
        "oracle_tau": oracle,
        "nonprivate_oracle_tau": nonprivate,
        "gap_to_nonprivate_game": gap,
        "gap_is_a_change_of_game_not_a_variance_term": True,
        "ipw_forecast": forecast,
        "ipw_b0": zero,
        "inclusion_exclusion": ie,
        "linear_regression": ols,
        "linear_regression_population_target": "participation_effect_because_independent_bernoulli",
        "round_fe_identified": False,
        "round_fe_gram_max_abs": diag["gram_max_abs"],
        "round_fe_failure_reason": diag["failure_reason"],
        "exact_population_variance_split_b0_enumerated": split,
        "variance_split_is_not_a_sample_certificate": True,
        "mae_ipw_forecast": mean_abs_error(forecast, oracle),
        "mae_ipw_b0": mean_abs_error(zero, oracle),
        "mae_inclusion_exclusion": mean_abs_error(ie, oracle),
        "mae_linear_regression": mean_abs_error(ols, oracle),
        "spearman_ipw_forecast": spearman(forecast, oracle),
        "sign_hits_ipw_forecast": int(np.sum(np.sign(forecast) == np.sign(oracle))),
        "conservative_freedman_halfwidth": freedman_halfwidth(t_rounds, setting.kappa, PREDECLARED_DELTA),
        "plugin_is_not_certified": True,
    }
    if setting.quorum is not None and n_aborted <= 0:
        raise RuntimeError("quorum setting dropped or never aborted; aborted draws must be included")
    return rows, _jsonable(summary)


def expanding_check(d: np.ndarray) -> np.ndarray:
    from .estimators import expanding_baseline

    return expanding_baseline(d)


def _one_fixedk(protocol, paper_flag, seed, t_rounds, k: int) -> tuple[list[dict], dict]:
    rng = _rng(seed, 100 + k)
    z = draw_fixed_size(rng, t_rounds, k, sigma=0.0)
    # Separate stream for the utility noise slot; sigma is 0 so this draw is unused.
    d = fixed_size_outcomes(z, sigma=0.0, rng=rng)
    if not np.allclose(z.sum(axis=1), k):
        raise RuntimeError("fixed-size draw did not have size k")
    repl = replacement_effects(k, sigma=0.0)
    add_half = participation_effects(0.5, sigma=0.0, quorum=None)
    add_rate = participation_effects(k / N_CLIENTS, sigma=0.0, quorum=None)
    ie, _ = inclusion_exclusion(z, d)
    repl_rank = ranks_descending(repl)
    est_rank = ranks_descending(ie)
    mae = mean_abs_error(ie, repl)
    corr = spearman(ie, repl)
    rows = []
    for i in range(N_CLIENTS):
        est_i = float(ie[i])
        rows.append(
            {
                "protocol": protocol,
                "paper_protocol": int(paper_flag),
                "seed": int(seed),
                "T": int(t_rounds),
                "k": int(k),
                "n_clients": N_CLIENTS,
                "client_id": i,
                "client_role": CLIENT_ROLES[i],
                "estimand": "fixed_size_replacement",
                "replacement_oracle": float(repl[i]),
                "replacement_sign": sign_of(float(repl[i])),
                "replacement_rank": repl_rank[i],
                "estimate_inclusion_exclusion": est_i,
                "estimate_sign": sign_of(est_i),
                "estimate_rank": est_rank[i],
                "abs_error_vs_replacement": abs(est_i - float(repl[i])),
                "sign_match_replacement": int(sign_of(est_i) == sign_of(float(repl[i]))),
                "bernoulli_addone_oracle_p_half": float(add_half[i]),
                "bernoulli_addone_oracle_at_marginal_rate": float(add_rate[i]),
                "abs_gap_replacement_minus_addone_p_half": abs(float(repl[i] - add_half[i])),
                "not_the_add_one_estimand": 1,
                "setting_mae_vs_replacement": mae,
                "setting_spearman_vs_replacement": corr,
                "draws_dropped": 0,
            }
        )
    summary = {
        "k": k,
        "seed": int(seed),
        "T": int(t_rounds),
        "estimand": "fixed_size_replacement",
        "not_the_add_one_estimand": True,
        "replacement_oracle": repl,
        "estimate_inclusion_exclusion": ie,
        "bernoulli_addone_oracle_p_half": add_half,
        "bernoulli_addone_oracle_at_marginal_rate": add_rate,
        "mae_vs_replacement": mae,
        "mean_abs_gap_vs_addone_p_half": float(np.mean(np.abs(repl - add_half))),
    }
    return rows, _jsonable(summary)


def run_protocol(
    protocol: str,
    outdir: Path,
    execution_host: str = "local",
    t_rounds: int | None = None,
    seeds: tuple[int, ...] | None = None,
) -> dict:
    """Write the Bernoulli ledger and the separate fixed-size ledger."""
    default_t, default_seeds, paper_flag = _protocol_budget(protocol)
    if seeds is None:
        seeds = default_seeds
    else:
        seeds = tuple(int(seed) for seed in seeds)
    if protocol == "paper" and t_rounds not in (None, PAPER_T):
        raise ValueError("protocol paper is locked at T=4000; use paper_large to change T")
    if protocol == "paper_large":
        if t_rounds is None:
            raise ValueError("paper_large requires T")
        if int(t_rounds) < PAPER_T:
            raise ValueError(f"paper_large T={t_rounds} is shorter than the locked paper budget {PAPER_T}")
        if seeds != tuple(PAPER_SEEDS):
            raise ValueError("paper_large uses seeds 0 1 2 3 4")
    if t_rounds is None:
        if default_t is None:
            raise ValueError(f"{protocol} requires T")
        t_rounds = default_t
    t_rounds = int(t_rounds)
    if t_rounds < 1:
        raise ValueError("T must be positive")
    check_status = run_checks()
    bernoulli_rows: list[dict] = []
    fixed_rows: list[dict] = []
    setting_summaries = []
    fixed_summaries = []
    for setting in bernoulli_settings():
        for seed in seeds:
            rows, summary = _one_bernoulli(protocol, paper_flag, seed, t_rounds, setting)
            bernoulli_rows.extend(rows)
            setting_summaries.append(summary)
            print(
                f"SETTING_DONE setting={setting.name} p={setting.p} seed={seed} T={t_rounds}",
                flush=True,
            )
    for k in (2, 3):
        for seed in seeds:
            rows, summary = _one_fixedk(protocol, paper_flag, seed, t_rounds, k)
            fixed_rows.extend(rows)
            fixed_summaries.append(summary)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    write_csv(outdir / "experiment_results.csv", BERNOULLI_COLUMNS, bernoulli_rows)
    write_csv(outdir / "fixedk_replacement_results.csv", FIXEDK_COLUMNS, fixed_rows)
    write_dictionary(outdir / "README.md", BERNOULLI_COLUMNS, "Column dictionary")
    # The fixed-k dictionary is the second half of a reader surprise if merged.
    # Write it as its own file next to that CSV.
    fixed_dict = ["# Fixed-size replacement ledger", "", "This file is not the Bernoulli participation-effect table.", ""]
    for name, desc in FIXEDK_COLUMNS:
        fixed_dict.append(f"- `{name}`: {desc}")
    fixed_dict.append("")
    (outdir / "FIXEDK_COLUMNS.md").write_text("\n".join(fixed_dict), encoding="utf-8")

    tau_half = participation_effects(0.5, sigma=0.0, quorum=None)
    tau_policy = participation_effects(0.3, sigma=0.0, quorum=None)
    from .oracle import partner_conditional_effect

    win_bar = evaluate_win_bar(bernoulli_rows) if paper_flag else {
        "win_bar_pass": False,
        "not_scored": "smoke is not the paper protocol",
        "plugin_is_not_certified": True,
        "halfwidth_formula": HALFWIDTH_FORMULA,
    }
    summary = {
        "protocol": protocol,
        "paper_protocol": paper_flag,
        "smoke_is_not_the_paper_protocol": not paper_flag,
        "run_definition": {
            "name": protocol,
            "T": int(t_rounds),
            "seeds": list(seeds),
            "delta": PREDECLARED_DELTA,
            "kappa_rule": "min(p, 1-p)",
            "halfwidth_formula": HALFWIDTH_FORMULA,
            "v_bound": "T*(4/kappa)**2",
            "abs_psi_minus_tau_bound": "4/kappa",
            "game_id": GAME_ID,
            "description": (
                "paper_large lengthens the same sigmoid_latent_v1 window. "
                "It is not the smoke budget and it does not replace the T=4000 paper ledger."
                if protocol == "paper_large"
                else "Named protocol budget. Smoke is not a paper protocol."
            ),
        },
        "paper_protocol_definition": {
            "name": "paper",
            "T": PAPER_T,
            "seeds": list(PAPER_SEEDS),
            "description": "Locked T=4000 five-seed ledger. Not the smoke budget. paper_large is a longer window of the same game.",
        },
        "smoke_protocol_definition": {"name": "smoke", "T": SMOKE_T, "seeds": list(SMOKE_SEEDS)},
        "win_bar_evaluation": win_bar,
        "execution_host": execution_host,
        "game_id": GAME_ID,
        "main_effects": MAIN,
        "gamma_complement_0_and_1": GAMMA_01,
        "dp_sigma_when_on": DP_SIGMA,
        "utility": "logistic of the latent aggregate, in (0, 1); D = u(after) - logistic(0)",
        "estimator": "A=(Z-p)/(p(1-p)), psi=A(D-b), tauhat=mean_t psi",
        "baseline": "expanding mean of previous releases, plus a b=0 ablation",
        "unit_checks": check_status,
        "oracle_nonprivate_p_half": tau_half,
        "oracle_nonprivate_p_0_3": tau_policy,
        "p_half_is_uniform_banzhaf_of_the_stated_game": True,
        "p_0_3_is_not_uniform_banzhaf": True,
        "complement_effect_of_client_1_given_client_0_in": partner_conditional_effect(1, 0, 1, 0.5, 0.0),
        "complement_effect_of_client_1_given_client_0_out": partner_conditional_effect(1, 0, 0, 0.5, 0.0),
        "dp_noise_is_not_mean_zero_utility_noise": True,
        "round_fixed_effect_regression_is_a_failing_diagnostic_not_the_estimator": True,
        "independent_bernoulli_ols_equals_participation_effect": True,
        "ols_does_not_recover_the_full_coalition_function": True,
        "sample_variance_plugin_is_not_certified": True,
        "fixed_k_is_a_separate_file": "fixedk_replacement_results.csv",
        "full_compliance_assumed": True,
        "dropout_not_simulated": True,
        "time_varying_state_not_simulated": False,
        "frozen_settings_remain_frozen": True,
        "drift_is_an_extra_setting": True,
        "drift_setting": DRIFT_SETTING_NAME,
        "drift_amplitude": DRIFT_AMPLITUDE,
        "drift_path": "amplitude * sin(2*pi*t/T) added to the latent intercept; same game_id",
        "state_is_frozen_so_the_path_average_equals_the_state_effect": False,
        "settings": setting_summaries,
        "fixed_k": fixed_summaries,
        "n_bernoulli_rows": len(bernoulli_rows),
        "n_fixedk_rows": len(fixed_rows),
    }
    summary = _jsonable(summary)
    write_json(outdir / "experiment_summary.json", summary)
    protocol_note = [
        "# Protocol ledger",
        "",
        f"- protocol: `{protocol}`",
        f"- T: {t_rounds}",
        f"- seeds: {list(seeds)}",
        f"- delta: {PREDECLARED_DELTA}",
        "- kappa: min(p, 1-p)",
        f"- halfwidth formula: {HALFWIDTH_FORMULA}",
        f"- game_id: {GAME_ID}",
        "- plugin half-width is not certified",
        "- smoke is not this paper protocol when protocol is paper or paper_large",
        "",
    ]
    (outdir / "PROTOCOL_LEDGER.md").write_text("\n".join(protocol_note), encoding="utf-8")
    write_json(
        outdir / "fixedk_replacement_summary.json",
        {
            "protocol": protocol,
            "paper_protocol": paper_flag,
            "estimand": "fixed_size_replacement",
            "not_mixed_into": "experiment_results.csv",
            "blocks": fixed_summaries,
        },
    )
    return summary
