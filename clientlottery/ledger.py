"""CSV ledger columns. Descriptions are written beside each results directory."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

BERNOULLI_COLUMNS: list[tuple[str, str]] = [
    ("protocol", "smoke, paper (T=4000), or paper_large (T named in the JSON). Smoke rows are not a paper protocol."),
    ("paper_protocol", "1 if this row used the paper budget (T and seeds in the JSON). 0 for smoke."),
    ("seed", "Numpy SeedSequence stream id for this setting."),
    ("T", "Number of rounds in the window. Aborted quorum rounds are included."),
    ("window_w", "Denominator of the averaged estimator. Equal to T."),
    ("n_clients", "Cohort size. Fixed at 5."),
    ("frozen_state", "1 if every round is redrawn at the same pre-round state."),
    ("kappa", "Floor used only in the conservative radius. Draws are Bernoulli(p), not a rejection sample. kappa = min(p, 1-p)."),
    ("p", "Known inclusion probability. Constant inside a row."),
    ("policy_name", "p_half_uniform_banzhaf_of_stated_game only when p=1/2. Otherwise policy_weighted_not_uniform_banzhaf."),
    ("setting", "frozen_nonprivate, frozen_dp, frozen_quorum_noop, or the extra drift_nonprivate path."),
    ("game_id", "Coefficient id of the synthetic logistic game."),
    ("dp_sigma", "Std. dev. of Gaussian noise added to the latent state before the logistic. 0 means the nonprivate game."),
    ("quorum", "Minimum included clients. Empty if aggregation always runs."),
    ("quorum_label", "public_noop_below_quorum_effect_includes_enabling when aborted draws return D=0 and stay in the average."),
    ("estimator", "ipw_forecast, ipw_b0, inclusion_exclusion, linear_regression, or round_fe_saturated."),
    ("targets_participation_effect", "1 if the population target is the policy-weighted participation effect. Under independent Bernoulli participation this includes IPW, inclusion/exclusion, and linear regression: the least-squares coefficient equals that effect for any coalition utility. 0 only for saturated round effects, which are unidentified."),
    ("client_id", "0 helpful, 1 complement of 0, 2 harmful, 3 moderate positive, 4 weak negative."),
    ("client_role", "Role name locked to client_id."),
    ("oracle_tau", "Enumerated participation effect of the stated game (16 coalitions of the other clients)."),
    ("oracle_sign", "Sign of oracle_tau. 1, 0, or -1."),
    ("oracle_rank", "1 = largest oracle effect. Ties share the minimum rank."),
    ("nonprivate_oracle_tau", "Enumerated effect of the nonprivate game at the same p, with no quorum."),
    ("gap_to_nonprivate_game", "oracle_tau minus nonprivate_oracle_tau. For the DP rows this is a change of game, not a variance term."),
    ("estimate", "Estimator output. Empty when the functional is unidentified on this transcript."),
    ("estimate_sign", "Sign of estimate."),
    ("estimate_rank", "1 = largest estimate within estimator and setting. Empty if the estimate is empty."),
    ("abs_error", "Absolute error against oracle_tau of the stated game."),
    ("sign_match", "1 if the estimate sign matches the oracle sign."),
    ("identified", "1 if this transcript uniquely determined the reported functional. 0 for saturated round effects and for empty inclusion cells."),
    ("realized_binary_variation", "1 if client i is both in and out at least once."),
    ("failure_reason", "Why identified=0. Empty when the functional was computable."),
    ("design_rank", "Rank of the design used by this estimator. For round effects, rank of [round dummies | participation]."),
    ("design_ncols", "Column count of that design, counting the intercept for linear regression and the round dummies for round effects."),
    ("gram_max_abs", "Max absolute entry of the participation Gram matrix after residualizing round effects. Zero when those effects saturate."),
    ("rank_check", "numerical_svd when T<=256, otherwise algebraic_one_observation_per_round. Empty for estimators that are not the round-effect diagnostic."),
    ("conservative_freedman_halfwidth", "Note's radius with v = T*(4/kappa)^2 and the predeclared delta. IPW rows only. Not a sample-variance certificate."),
    ("covered_by_conservative_freedman", "1 if abs_error is at most the conservative radius. IPW rows only."),
    ("bonferroni_freedman_halfwidth", "Same radius with delta/n_clients. Predeclared simultaneous control over the five clients, not a time-uniform bound."),
    ("covered_by_bonferroni_freedman", "1 if abs_error is at most the Bonferroni radius. IPW rows only."),
    ("plugin_halfwidth_not_certified", "Same algebra with v = T times the sample variance of psi. Not certified."),
    ("plugin_covered_not_certified", "Coverage of that plug-in radius. Not a certificate."),
    ("predeclared_delta", "Marginal tail level used in the conservative radius. 0.05."),
    ("inclusion_count", "Rounds with Z_i=1. Aborted rounds are counted."),
    ("exclusion_count", "Rounds with Z_i=0."),
    ("effective_overlap_setting", "Minimum over clients of min(inclusion rate, exclusion rate)."),
    ("n_aborted_included", "Quorum failures kept in the average. 0 when the setting has no quorum."),
    ("draws_dropped", "Always 0. Aborted draws are not removed."),
    ("max_abs_D", "Max |D_t| in the transcript. The conservative radius uses |D|<=1."),
    ("setting_mae", "Mean absolute error of this estimator against oracle_tau, over clients with a numeric estimate."),
    ("setting_spearman", "Spearman correlation of estimates with oracle_tau. Empty if fewer than three numeric estimates."),
    ("setting_sign_hits", "How many clients matched the oracle sign."),
    ("b_rule", "expanding_mean_past_release, zero, or not_applicable."),
]

FIXEDK_COLUMNS: list[tuple[str, str]] = [
    ("protocol", "smoke or paper. Not mixed into experiment_results.csv."),
    ("paper_protocol", "1 for the paper budget, 0 for smoke."),
    ("seed", "Seed stream id."),
    ("T", "Number of fixed-size draws."),
    ("k", "Exact coalition size each round."),
    ("n_clients", "Cohort size. Fixed at 5."),
    ("client_id", "Same ids as the Bernoulli ledger."),
    ("client_role", "Role name locked to client_id."),
    ("estimand", "fixed_size_replacement. Not the Bernoulli add-one effect."),
    ("replacement_oracle", "Exact E[D|Z_i=1] - E[D|Z_i=0] under uniform subsets of size k."),
    ("replacement_sign", "Sign of the replacement oracle."),
    ("replacement_rank", "1 = largest replacement effect."),
    ("estimate_inclusion_exclusion", "Inclusion/exclusion contrast on the fixed-size transcript."),
    ("estimate_sign", "Sign of that contrast."),
    ("estimate_rank", "1 = largest contrast."),
    ("abs_error_vs_replacement", "Absolute error against the replacement oracle."),
    ("sign_match_replacement", "1 if the contrast sign matches the replacement oracle."),
    ("bernoulli_addone_oracle_p_half", "Add-one effect under independent Bernoulli(1/2). A different estimand."),
    ("bernoulli_addone_oracle_at_marginal_rate", "Add-one effect under independent Bernoulli(k/n). Still not the fixed-size contrast."),
    ("abs_gap_replacement_minus_addone_p_half", "replacement oracle minus Bernoulli(1/2) add-one oracle."),
    ("not_the_add_one_estimand", "Always 1 in this file."),
    ("setting_mae_vs_replacement", "Mean absolute error against the replacement oracle."),
    ("setting_spearman_vs_replacement", "Spearman correlation with the replacement oracle."),
    ("draws_dropped", "Always 0."),
]


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "1" if bool(value) else "0"
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(float(value)):
            return ""
        return f"{float(value):.8g}"
    return str(value)


def write_csv(path: Path, columns: list[tuple[str, str]], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [name for name, _ in columns]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        for row in rows:
            extra = set(row) - set(names)
            missing = [name for name in names if name not in row]
            if extra or missing:
                raise KeyError(f"extra={sorted(extra)} missing={missing}")
            writer.writerow({name: _cell(row.get(name)) for name in names})


def write_dictionary(path: Path, columns: list[tuple[str, str]], title: str) -> None:
    lines = [
        f"# {title}",
        "",
        "Smoke rows are a smoke budget. They are not the paper protocol.",
        "The plug-in half-width is not a certificate. The conservative radius uses the deterministic bound v = T*(4/kappa)^2.",
        "Fixed-size sampling, when present, lives in `fixedk_replacement_results.csv` and is a replacement effect.",
        "",
    ]
    for name, desc in columns:
        lines.append(f"- `{name}`: {desc}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
