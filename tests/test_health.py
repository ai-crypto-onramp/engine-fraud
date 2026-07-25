from httpx import ASGITransport, AsyncClient

from fraud_engine.app import READINESS_CHECKS, app, classify_readiness, readiness_report


async def test_healthz() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_readyz_ok() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/readyz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["healthy"] == str(len(READINESS_CHECKS))
    assert body["failed"] == "0"
    assert body["total"] == str(len(READINESS_CHECKS))
    for name, _ in READINESS_CHECKS:
        assert body[name] == "ok"


def test_readiness_report_all_ok() -> None:
    results, failed, total = readiness_report()
    assert failed == 0
    assert total == len(READINESS_CHECKS)
    assert results["db"] == "ok"


def test_readiness_report_with_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        "fraud_engine.app.READINESS_CHECKS",
        [("db", lambda: True), ("mq", lambda: False)],
    )
    results, failed, total = readiness_report()
    assert failed == 1
    assert total == 2
    assert results["db"] == "ok"
    assert results["mq"] == "down"


def test_classify_readiness() -> None:
    assert classify_readiness(0, 18) == (200, "ready")
    assert classify_readiness(3, 18) == (200, "degraded")
    assert classify_readiness(0, 0) == (200, "ready")
