from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def psi(expected: Sequence[float], actual: Sequence[float], bins: int = 10, eps: float = 1e-6) -> float:
    exp = np.asarray(expected, dtype=float)
    act = np.asarray(actual, dtype=float)
    if exp.size == 0 or act.size == 0:
        return 0.0
    lo = float(min(exp.min(), act.min()))
    hi = float(max(exp.max(), act.max()))
    if hi <= lo:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    exp_counts, _ = np.histogram(exp, bins=edges)
    act_counts, _ = np.histogram(act, bins=edges)
    exp_pct = exp_counts / max(exp_counts.sum(), 1)
    act_pct = act_counts / max(act_counts.sum(), 1)
    exp_pct = np.where(exp_pct <= 0, eps, exp_pct)
    act_pct = np.where(act_pct <= 0, eps, act_pct)
    terms = (act_pct - exp_pct) * np.log(act_pct / exp_pct)
    return float(np.sum(terms))


def ks_statistic(expected: Sequence[float], actual: Sequence[float]) -> float:
    exp = np.sort(np.asarray(expected, dtype=float))
    act = np.sort(np.asarray(actual, dtype=float))
    if exp.size == 0 or act.size == 0:
        return 0.0
    all_vals = np.concatenate([exp, act])
    cdf_exp = np.searchsorted(exp, all_vals, side="right") / exp.size
    cdf_act = np.searchsorted(act, all_vals, side="right") / act.size
    return float(np.max(np.abs(cdf_exp - cdf_act)))


def drift_report(
    baseline: Mapping[str, Sequence[float]], current: Mapping[str, Sequence[float]],
    *, psi_threshold: float, bins: int = 10,
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for feature, cur in current.items():
        base = baseline.get(feature)
        if base is None or len(base) == 0 or len(cur) == 0:
            continue
        p = psi(base, cur, bins=bins)
        k = ks_statistic(base, cur)
        out.append({
            "feature": feature,
            "psi": round(p, 5),
            "ks": round(k, 5),
            "breached": bool(p > psi_threshold),
        })
    return out


def breached_features(report: list[dict[str, object]]) -> list[str]:
    return [str(r["feature"]) for r in report if r["breached"]]
