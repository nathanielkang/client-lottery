"""Predeclared manuscript win bar for the paper_large ledger.

The conservative half-width is a function of T, kappa, and delta only.
A sample-variance plug-in is never treated as certified.
"""

from __future__ import annotations

import csv
from pathlib import Path

HELPFUL = "helpful_main"
HARMFUL = "harmful_main"
COMPLEMENT = "complement_of_0"
MODERATE = "moderate_positive"

HALFWIDTH_FORMULA = (
    "(sqrt(2*v*log(2/delta)) + (2/3)*(4/kappa)*log(2/delta))/T "
    "with v = T*(4/kappa)^2, kappa = min(p, 1-p), |psi-tau| <= 4/kappa"
)

PREDECLARED_ITEMS = {
    "1_certifiable_large_effects": (
        "At p=0.5, frozen_nonprivate and frozen_dp, helpful_main and harmful_main "
        "satisfy |oracle_tau| > conservative_freedman_halfwidth and sign_match=1 "
        "on every seed. complement_of_0 is recorded when the same inequality holds; "
        "it is preferred and is not required for item 1."
    ),
    "2_point_recovery": (
        "ipw_forecast setting_mae <= 0.02 and setting_spearman = 1 on every seed "
        "for frozen_nonprivate and frozen_dp at p=0.5."
    ),
    "3_ranking": (
        "Predeclared pair helpful_main and harmful_main on frozen_nonprivate at p=0.5: "
        "oracle separation exceeds 2 times the conservative half-width, and the "
        "ipw_forecast estimates keep helpful above harmful on every seed."
    ),
    "4_policy_flip": (
        "On frozen_nonprivate, complement_of_0 versus moderate_positive flips between "
        "p=0.5 and p=0.3 in the oracle, and ipw_forecast recovers that flip on at least 4 of 5 seeds."
    ),
    "5_round_fe": (
        "round_fe_saturated stays unidentified (identified=0) on every paper-transcript row."
    ),
    "6_quorum_signs": (
        "frozen_quorum_noop ipw_forecast matches the quorum oracle sign for helpful_main "
        "and harmful_main on at least 4 of 5 seeds."
    ),
    "plugin": "plugin_halfwidth_not_certified is descriptive only and is not a certificate.",
}


def _f(value) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def _i(value) -> int:
    return int(float(value))


def load_bernoulli_csv(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _ipw(rows: list[dict], setting: str, p: float) -> list[dict]:
    out = []
    for row in rows:
        if row.get("estimator") != "ipw_forecast":
            continue
        if row.get("setting") != setting:
            continue
        if abs(_f(row.get("p")) - p) > 1e-9:
            continue
        out.append(row)
    return out


def _role(rows: list[dict], role: str) -> list[dict]:
    return [row for row in rows if row.get("client_role") == role]


def _seeds(rows: list[dict]) -> list[int]:
    return sorted({_i(row["seed"]) for row in rows})


def _certified(role_rows: list[dict]) -> bool:
    if len(_seeds(role_rows)) < 5 or len(role_rows) < 5:
        return False
    for row in role_rows:
        halfwidth = _f(row.get("conservative_freedman_halfwidth"))
        if not (abs(_f(row.get("oracle_tau"))) > halfwidth):
            return False
        if _i(row.get("sign_match")) != 1:
            return False
    return True


def _point_ok(rows: list[dict]) -> bool:
    seen = {}
    for row in rows:
        if row.get("client_role") != HELPFUL:
            continue
        seen[_i(row["seed"])] = row
    if len(seen) < 5:
        return False
    for row in seen.values():
        if _f(row.get("setting_mae")) > 0.02 + 1e-12:
            return False
        if _f(row.get("setting_spearman")) < 1.0 - 1e-9:
            return False
    return True


def _pair_order(rows: list[dict]) -> tuple[bool, float, float]:
    """Return (order preserved and separation clears 2*halfwidth, separation, halfwidth)."""
    by_seed: dict[int, dict[str, dict]] = {}
    for row in rows:
        if row.get("client_role") not in (HELPFUL, HARMFUL):
            continue
        by_seed.setdefault(_i(row["seed"]), {})[row["client_role"]] = row
    if len(by_seed) < 5:
        return False, float("nan"), float("nan")
    separation = float("nan")
    halfwidth = float("nan")
    for pair in by_seed.values():
        if HELPFUL not in pair or HARMFUL not in pair:
            return False, separation, halfwidth
        helpful = pair[HELPFUL]
        harmful = pair[HARMFUL]
        separation = abs(_f(helpful["oracle_tau"]) - _f(harmful["oracle_tau"]))
        halfwidth = _f(helpful["conservative_freedman_halfwidth"])
        if not (separation > 2.0 * halfwidth):
            return False, separation, halfwidth
        if _f(helpful["estimate"]) <= _f(harmful["estimate"]):
            return False, separation, halfwidth
    return True, separation, halfwidth


def _flip_hits(nonprivate_half: list[dict], nonprivate_low: list[dict]) -> tuple[int, bool]:
    def gap(rows: list[dict], seed: int) -> tuple[float, float]:
        comp = next(row for row in rows if _i(row["seed"]) == seed and row["client_role"] == COMPLEMENT)
        mod = next(row for row in rows if _i(row["seed"]) == seed and row["client_role"] == MODERATE)
        oracle_gap = _f(comp["oracle_tau"]) - _f(mod["oracle_tau"])
        estimate_gap = _f(comp["estimate"]) - _f(mod["estimate"])
        return oracle_gap, estimate_gap

    seeds = _seeds(nonprivate_half)
    if len(seeds) < 5:
        return 0, False
    oracle_flip = False
    hits = 0
    for seed in seeds:
        o_hi, e_hi = gap(nonprivate_half, seed)
        o_lo, e_lo = gap(nonprivate_low, seed)
        oracle_flip = o_hi * o_lo < 0
        if oracle_flip and e_hi * o_hi > 0 and e_lo * o_lo > 0:
            hits += 1
    return hits, oracle_flip


def evaluate_win_bar(rows: list[dict]) -> dict:
    """Score items 1–6. Does not treat the plug-in width as certified."""
    half = _ipw(rows, "frozen_nonprivate", 0.5)
    low = _ipw(rows, "frozen_nonprivate", 0.3)
    dp = _ipw(rows, "frozen_dp", 0.5)
    quorum = _ipw(rows, "frozen_quorum_noop", 0.5)
    fe_rows = [row for row in rows if row.get("estimator") == "round_fe_saturated"]

    required_roles = {}
    complement = {}
    for label, block in (("frozen_nonprivate_p_0.5", half), ("frozen_dp_p_0.5", dp)):
        required_roles[label] = {
            HELPFUL: _certified(_role(block, HELPFUL)),
            HARMFUL: _certified(_role(block, HARMFUL)),
        }
        complement[label] = _certified(_role(block, COMPLEMENT))

    item1 = all(ok for block in required_roles.values() for ok in block.values())
    item2 = _point_ok(half) and _point_ok(dp)
    item3, separation, halfwidth = _pair_order(half)
    flip_hits, oracle_flip = _flip_hits(half, low) if half and low else (0, False)
    item4 = bool(oracle_flip) and flip_hits >= 4
    item5 = bool(fe_rows) and all(_i(row.get("identified")) == 0 for row in fe_rows)
    quorum_hits = 0
    quorum_seeds = _seeds(quorum)
    for seed in quorum_seeds:
        ok = True
        for role in (HELPFUL, HARMFUL):
            match = [
                row
                for row in quorum
                if _i(row["seed"]) == seed and row.get("client_role") == role and _i(row.get("sign_match")) == 1
            ]
            ok = ok and bool(match)
        quorum_hits += int(ok)
    item6 = quorum_hits >= 4 and len(quorum_seeds) >= 5

    items = {
        "1_certifiable_large_effects": item1,
        "2_point_recovery": item2,
        "3_ranking": item3,
        "4_policy_flip": item4,
        "5_round_fe": item5,
        "6_quorum_signs": item6,
    }
    return {
        "predeclared_win_bar": PREDECLARED_ITEMS,
        "halfwidth_formula": HALFWIDTH_FORMULA,
        "plugin_is_not_certified": True,
        "items_pass": items,
        "win_bar_pass": all(items.values()),
        "required_clients_certified": required_roles,
        "complement_also_certified": complement,
        "ranking_pair": "helpful_main_above_harmful_main",
        "ranking_setting": "frozen_nonprivate_p_0.5",
        "oracle_separation_helpful_vs_harmful": separation,
        "conservative_halfwidth_at_ranking_setting": halfwidth,
        "policy_flip_oracle": oracle_flip,
        "policy_flip_ipw_seeds": flip_hits,
        "quorum_sign_seeds": quorum_hits,
        "round_fe_rows": len(fe_rows),
    }
