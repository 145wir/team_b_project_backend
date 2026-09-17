from __future__ import annotations

import os

from fastapi.testclient import TestClient


# ============================================================
# Test environment
# ============================================================
#
# CI나 .env가 없는 환경에서도
# app.core.config.Settings 생성 자체가 실패하지 않도록
# 테스트용 기본값을 넣는다.
#
# 실제 DB에는 연결하지 않는다.
# ============================================================

os.environ.setdefault(
    "DB_HOST",
    "localhost",
)

os.environ.setdefault(
    "DB_PORT",
    "5432",
)

os.environ.setdefault(
    "DB_NAME",
    "team_b_ai",
)

os.environ.setdefault(
    "DB_USER",
    "test_user",
)

os.environ.setdefault(
    "DB_PASSWORD",
    "test_password",
)

os.environ.setdefault(
    "DB_SSLMODE",
    "require",
)

os.environ.setdefault(
    "RETRIEVER_MODE",
    "mock",
)


from app.main import app  # noqa: E402
import app.api.health as health_api  # noqa: E402


client = TestClient(
    app
)


# ============================================================
# Success
# ============================================================


def test_health_returns_200_when_database_is_available(
    monkeypatch,
) -> None:
    """
    DB 연결이 정상이라고 가정했을 때:

        GET /api/v1/health

    가 HTTP 200과 정상적인 JSON을 반환하는지 검사한다.

    실제 AWS RDS에는 접속하지 않는다.
    """

    def fake_check_database_connection() -> dict:
        return {
            "database_name": "team_b_ai",
            "schema_name": "public",
        }

    monkeypatch.setattr(
        health_api,
        "check_database_connection",
        fake_check_database_connection,
    )

    response = client.get(
        "/api/v1/health"
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["status"] == "ok"

    assert payload["database"]["status"] == (
        "connected"
    )

    assert payload["database"]["name"] == (
        "team_b_ai"
    )

    assert payload["database"]["schema"] == (
        "public"
    )

    assert "application" in payload
    assert "environment" in payload


# ============================================================
# Failure
# ============================================================


def test_health_returns_503_when_database_is_unavailable(
    monkeypatch,
) -> None:
    """
    DB 연결이 실패했다고 가정했을 때:

        GET /api/v1/health

    가 HTTP 503을 반환하는지 검사한다.
    """

    def fake_check_database_connection() -> dict:
        raise RuntimeError(
            "mock database failure"
        )

    monkeypatch.setattr(
        health_api,
        "check_database_connection",
        fake_check_database_connection,
    )

    response = client.get(
        "/api/v1/health"
    )

    assert response.status_code == 503

    payload = response.json()

    assert "detail" in payload

    assert (
        "Database connection failed"
        in payload["detail"]
    )

    assert (
        "mock database failure"
        in payload["detail"]
    )