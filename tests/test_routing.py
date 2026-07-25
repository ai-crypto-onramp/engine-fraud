from fraud_engine.models.routing import pick_variant, resolve_split, risk_band, stable_hash_unit


def test_risk_band_thresholds() -> None:
    assert risk_band(0.8, 0.75, 0.40) == "HIGH"
    assert risk_band(0.5, 0.75, 0.40) == "MEDIUM"
    assert risk_band(0.1, 0.75, 0.40) == "LOW"


def test_stable_hash_is_deterministic() -> None:
    assert stable_hash_unit("tx_1") == stable_hash_unit("tx_1")
    assert 0.0 <= stable_hash_unit("tx_1") < 1.0


def test_pick_variant_force() -> None:
    assert pick_variant("tx1", challenger_fraction=0.5, force_variant="champion") == "champion"
    assert pick_variant("tx1", challenger_fraction=0.5, force_variant="challenger") == "challenger"


def test_pick_variant_split_edges() -> None:
    assert pick_variant("tx1", challenger_fraction=0.0) == "champion"
    assert pick_variant("tx1", challenger_fraction=1.0) == "challenger"


def test_pick_variant_distribution() -> None:
    counts = {"champion": 0, "challenger": 0}
    for i in range(10000):
        v = pick_variant(f"tx_{i}", challenger_fraction=0.25)
        counts[v] += 1
    frac = counts["challenger"] / 10000
    assert 0.24 <= frac <= 0.26


def test_pick_variant_same_tx_same_variant() -> None:
    a = pick_variant("tx_const", challenger_fraction=0.3)
    b = pick_variant("tx_const", challenger_fraction=0.3)
    assert a == b


def test_resolve_split_default() -> None:
    assert resolve_split(None, default_challenger_fraction=0.10) == 0.10
    assert resolve_split({"champion": 0.9, "challenger": 0.1}, default_challenger_fraction=0.10) == 0.1
