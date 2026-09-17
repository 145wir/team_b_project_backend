from __future__ import annotations

from app.core.config import settings
from app.schemas.retrieval import (
    RetrievalCandidate,
    RetrievalRequest,
    RetrieverResult,
)


# ============================================================
# Exceptions
# ============================================================


class RetrieverServiceError(
    RuntimeError
):
    """
    Retriever 실행 실패.
    """


class RetrieverTimeoutError(
    RetrieverServiceError
):
    """
    Retriever timeout.

    실제 RAG Core 연결 후 사용한다.
    """


# ============================================================
# Mock Retriever
# ============================================================


def _mock_retrieve_candidates(
    request: RetrievalRequest,
) -> RetrieverResult:
    """
    실제 RAG Core가 Backend에 통합되기 전까지
    HTTP End-to-End 파이프라인을 검증하기 위한
    개발용 Mock Retriever.

    절대 실제 검색 결과로 간주하지 않는다.
    """

    candidates = [
        RetrievalCandidate(
            retriever_rank=1,
            chunk_id=900001,
            document_id=9001,
            chunk_index=0,
            section_index=1,
            section_heading="Introduction",
            chunk_text=(
                "Deep-space spacecraft require increased "
                "onboard autonomy because long communication "
                "delays can prevent immediate intervention "
                "from ground operators."
            ),
            document_title=(
                "Autonomous Decision-Making "
                "for Deep-Space Missions"
            ),
            source="ntrs",
            source_id="MOCK-NTRS-001",
            topic_axis="onboard_ai",
            cosine_similarity=0.72,
            quality_score=0.95,
        ),

        RetrievalCandidate(
            retriever_rank=2,
            chunk_id=900002,
            document_id=9002,
            chunk_index=1,
            section_index=2,
            section_heading="Autonomous Operations",
            chunk_text=(
                "Onboard artificial intelligence can support "
                "spacecraft monitoring, planning, and autonomous "
                "decision-making when continuous communication "
                "with Earth is unavailable."
            ),
            document_title=(
                "Onboard Artificial Intelligence "
                "for Autonomous Spacecraft Operations"
            ),
            source="arxiv",
            source_id="MOCK-ARXIV-002",
            topic_axis="onboard_ai",
            cosine_similarity=0.69,
            quality_score=0.93,
        ),

        RetrievalCandidate(
            retriever_rank=3,
            chunk_id=900003,
            document_id=9003,
            chunk_index=2,
            section_index=3,
            section_heading="Fault Management",
            chunk_text=(
                "Autonomous fault detection and recovery "
                "can reduce dependence on real-time ground "
                "control during spacecraft operations."
            ),
            document_title=(
                "Autonomous Fault Management "
                "for Space Systems"
            ),
            source="ntrs",
            source_id="MOCK-NTRS-003",
            topic_axis="satellite_autonomy",
            cosine_similarity=0.64,
            quality_score=0.91,
        ),

        RetrievalCandidate(
            retriever_rank=4,
            chunk_id=900004,
            document_id=9004,
            chunk_index=0,
            section_index=1,
            section_heading="Navigation",
            chunk_text=(
                "Autonomous navigation allows robotic systems "
                "to make local decisions under uncertain "
                "environmental conditions."
            ),
            document_title=(
                "Autonomous Navigation "
                "in Uncertain Environments"
            ),
            source="arxiv",
            source_id="MOCK-ARXIV-004",
            topic_axis="rover_autonomy",
            cosine_similarity=0.57,
            quality_score=0.89,
        ),
    ]

    return RetrieverResult(
        title=request.title,
        research_idea=request.research_idea,
        candidate_count=len(
            candidates
        ),
        retriever_version=(
            "mock_retriever_v1"
        ),
        embedding_version=(
            "mock_svd256"
        ),
        vector_dimension=256,
        candidates=candidates,
    )


# ============================================================
# Real Retriever placeholder
# ============================================================


def _real_retrieve_candidates(
    request: RetrievalRequest,
) -> RetrieverResult:

    rows = rag_retrieve_candidates(
        title=request.title,
        research_idea=(
            request.research_idea
            or ""
        ),
        candidate_k=(
            settings.retrieval_candidate_k
        ),
    )

    candidates = [
        RetrievalCandidate(
            retriever_rank=row["rank"],
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            chunk_index=row["chunk_index"],
            section_index=row.get("section_index"),
            section_heading=row.get("section_heading"),
            chunk_text=row["chunk_text"],
            document_title=row["title"],
            source=row["source"],
            source_id=row["source_id"],
            topic_axis=row["topic_axis"],
            cosine_similarity=row["cosine_similarity"],
            quality_score=row.get("quality_score"),
        )
        for row in rows
    ]

    return RetrieverResult(
        title=request.title,
        research_idea=request.research_idea,
        candidate_count=len(candidates),
        retriever_version="실제 버전",
        embedding_version=(
            "core100_tfidf_svd_256_v1"
        ),
        vector_dimension=256,
        candidates=candidates,
    )

# ============================================================
# Public Interface
# ============================================================


def retrieve_candidates(
    request: RetrievalRequest,
) -> RetrieverResult:
    """
    Backend에서 사용하는 Retriever 단일 진입점.

    현재:
        RETRIEVER_MODE=mock

    향후:
        RETRIEVER_MODE=real
    """

    mode = (
        settings
        .retriever_mode
        .strip()
        .lower()
    )

    if mode == "mock":
        return (
            _mock_retrieve_candidates(
                request
            )
        )

    if mode == "real":
        return (
            _real_retrieve_candidates(
                request
            )
        )

    raise RetrieverServiceError(
        "Invalid RETRIEVER_MODE. "
        "Use 'mock' or 'real'. "
        f"Current value: {mode!r}"
    )