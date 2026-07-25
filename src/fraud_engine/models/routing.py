from __future__ import annotations

import hashlib


def risk_band(score: float, threshold_high: float, threshold_medium: float) -> str:
    if score >= threshold_high:
        return "HIGH"
    if score >= threshold_medium:
        return "MEDIUM"
    return "LOW"


def stable_hash_unit(tx_id: str, salt: str = "") -> float:
    h = hashlib.sha256((salt + tx_id).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / float(1 << 64)


def pick_variant(
    tx_id: str,
    *,
    challenger_fraction: float,
    champion_label: str = "champion",
    challenger_label: str = "challenger",
    force_variant: str | None = None,
    salt: str = "",
) -> str:
    if force_variant is not None:
        return force_variant
    if challenger_fraction <= 0.0:
        return champion_label
    if challenger_fraction >= 1.0:
        return challenger_label
    u = stable_hash_unit(tx_id, salt=salt)
    return challenger_label if u < challenger_fraction else champion_label


def resolve_split(
    traffic_split: dict[str, float] | None,
    *,
    default_challenger_fraction: float,
) -> float:
    if not traffic_split:
        return default_challenger_fraction
    return float(traffic_split.get("challenger", 0.0))
