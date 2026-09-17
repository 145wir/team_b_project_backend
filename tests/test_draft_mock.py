from __future__ import annotations

import os


# ============================================================
# Test environment
# ============================================================
#
# config.Settings가 import될 때
# 반드시 Mock Retriever 모드가 되도록
# app import 전에 환경변수를 설정한다.
#
# /draft Mock 테스트에서는 실제 AWS RDS에 접속하지 않는다.
# ============================================================

os.environ["RETRIEVER_MODE"] = "mock"

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


from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


client = TestClient(
    app
)


# ============================================================
# Test 1
# Draft Mock End-to-End
# ============================================================


def test_draft_mock_pipeline_returns_200() -> None:
    """
    Mock Retriever를 사용하는 전체 Draft Pipeline이

        MockRetriever
        -> Reranker
        -> MockGenerator
        -> Validator

    순서로 정상 실행되고 HTTP 200을 반환하는지 검사한다.
    """

    payload = {
        "title": (
            "Autonomous Onboard Decision-Making "
            "for Deep-Space Spacecraft "
            "Under Communication Delays"
        ),

        "research_idea": (
            "실시간 지상 통제가 어려운 심우주 환경에서 "
            "온보드 자율 의사결정과 인간 감독 사이의 "
            "균형을 연구한다."
        ),

        "writing_constraints": [
            "근거 없는 숫자 금지",
            "Evidence에 없는 사실 단정 금지",
            "과장 표현 금지",
        ],

        "language": "ko",

        "target_length": 4500,

        "top_k": 3,
    }

    response = client.post(
        "/api/v1/draft",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    # --------------------------------------------------------
    # Basic response
    # --------------------------------------------------------

    assert (
        data["title"]
        == payload["title"]
    )

    assert (
        data["language"]
        == "ko"
    )

    assert (
        data["target_length"]
        == 4500
    )


    # --------------------------------------------------------
    # Generator
    # --------------------------------------------------------

    assert "generator" in data

    assert (
        data["generator"]["generator_type"]
        == "MockGenerator"
    )

    assert (
        data["generator"]["is_mock"]
        is True
    )


    # --------------------------------------------------------
    # Sections
    # --------------------------------------------------------

    assert "sections" in data

    assert (
        len(
            data["sections"]
        )
        == 3
    )

    headings = [
        section["heading"]

        for section
        in data["sections"]
    ]

    assert headings == [
        "Introduction",
        "Body",
        "Conclusion",
    ]

    for section in data["sections"]:
        assert section["body"]
        assert len(
            section["body"]
        ) > 0


    # --------------------------------------------------------
    # Evidence
    # --------------------------------------------------------

    assert "evidence" in data

    assert (
        len(
            data["evidence"]
        )
        == 3
    )

    assert [
        item["rank"]

        for item
        in data["evidence"]
    ] == [
        1,
        2,
        3,
    ]

    for item in data["evidence"]:

        assert "chunk_id" in item

        assert "document_id" in item

        assert "chunk_text" in item

        assert "cosine_similarity" in item

        assert "rerank_score" in item

        assert (
            0.0
            <= item["semantic_score"]
            <= 1.0
        )

        assert (
            0.0
            <= item["lexical_score"]
            <= 1.0
        )

        assert (
            0.0
            <= item["title_overlap_score"]
            <= 1.0
        )


    # --------------------------------------------------------
    # References
    # --------------------------------------------------------

    assert "references" in data

    assert (
        len(
            data["references"]
        )
        >= 1
    )

    for reference in data["references"]:

        assert "document_id" in reference

        assert "document_title" in reference

        assert "source" in reference

        assert "source_id" in reference


    # --------------------------------------------------------
    # Validator
    # --------------------------------------------------------

    assert "validation" in data

    assert (
        data["validation"]
        is not None
    )

    assert (
        data["validation"]["validator_version"]
        == "rule_based_validator_v1"
    )

    assert "summary" in (
        data["validation"]
    )

    assert "issues" in (
        data["validation"]
    )

    assert (
        data["validation"]["summary"]["errors"]
        == 0
    )

    assert (
        data["validation"]["summary"]["passed"]
        is True
    )


# ============================================================
# Test 2
# Request validation
# ============================================================


def test_draft_rejects_invalid_request() -> None:
    """
    Pydantic DraftRequest validation이 실제 API에서도
    적용되는지 확인한다.

    title / research_idea가 너무 짧고,
    target_length가 허용 범위 미만이면
    HTTP 422가 나와야 한다.
    """

    payload = {
        "title": "A",

        "research_idea": "B",

        "writing_constraints": [],

        "language": "ko",

        "target_length": 100,

        "top_k": 3,
    }

    response = client.post(
        "/api/v1/draft",
        json=payload,
    )

    assert (
        response.status_code
        == 422
    )


# ============================================================
# Test 3
# Top-K contract
# ============================================================


def test_draft_respects_requested_top_k() -> None:
    """
    top_k=2 요청 시
    Reranker 최종 Evidence도 2개여야 한다.

    MockRetriever는 4개 Candidate를 만들지만,
    DraftResponse에는 Final Top-K만 전달되어야 한다.
    """

    payload = {
        "title": (
            "Autonomous Spacecraft "
            "Decision Making"
        ),

        "research_idea": (
            "통신 지연 환경에서 우주선의 "
            "자율 판단 구조를 연구한다."
        ),

        "writing_constraints": [],

        "language": "ko",

        "target_length": 4500,

        "top_k": 2,
    }

    response = client.post(
        "/api/v1/draft",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert (
        len(
            data["evidence"]
        )
        == 2
    )

    assert [
        item["rank"]

        for item
        in data["evidence"]
    ] == [
        1,
        2,
    ]


# ============================================================
# Test 4
# Mock identifiers
# ============================================================


def test_draft_uses_mock_retriever_data() -> None:
    """
    현재 자동 테스트는 반드시 Mock Retriever를 사용하는지
    확인한다.

    Real RAG 통합 이후에도 이 테스트는
    Mock Regression Test로 계속 유지할 수 있다.
    """

    payload = {
        "title": (
            "Deep Space Autonomous "
            "Operations"
        ),

        "research_idea": (
            "심우주 환경의 통신 제약과 "
            "온보드 자율성을 연구한다."
        ),

        "writing_constraints": [],

        "language": "ko",

        "target_length": 4500,

        "top_k": 2,
    }

    response = client.post(
        "/api/v1/draft",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["evidence"]

    for item in data["evidence"]:

        assert (
            item["source_id"]
            .startswith(
                "MOCK-"
            )
        )