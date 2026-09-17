from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


TopicAxis = Literal[
    "rover_autonomy",
    "onboard_ai",
    "satellite_autonomy",
]


# ============================================================
# Request
# ============================================================


class RetrievalRequest(BaseModel):
    """
    POST /api/v1/retrieve 요청.

    사용자는:
        - Research Title
        - Human Research Idea
        - 최종 Evidence 개수

    를 전달한다.

    Candidate 개수는 Backend 내부 설정으로 관리한다.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    title: str = Field(
        min_length=3,
        max_length=500,
        description="Research title",
    )

    research_idea: str | None = Field(
        default=None,
        max_length=3000,
        description="Human-provided research idea",
    )

    top_k: int = Field(
        default=6,
        ge=1,
        le=20,
        description="Final evidence count after reranking",
    )


# ============================================================
# Retriever candidate
# ============================================================


class RetrievalCandidate(BaseModel):
    """
    Frozen Vector Retriever가 반환한 Candidate.

    아직 최종 Evidence가 아니다.

    retriever_rank:
        Vector Retriever 단계의 원래 순위.

    cosine_similarity:
        query vector와 chunk vector 사이의 cosine similarity.
        정답 확률이 아니다.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    retriever_rank: int = Field(
        ge=1,
    )

    chunk_id: int = Field(
        ge=1,
    )

    document_id: int = Field(
        ge=1,
    )

    chunk_index: int = Field(
        ge=0,
    )

    section_index: int | None = Field(
        default=None,
        ge=0,
    )

    section_heading: str | None = None

    chunk_text: str = Field(
        min_length=1,
    )

    document_title: str = Field(
        min_length=1,
    )

    source: str = Field(
        min_length=1,
    )

    source_id: str = Field(
        min_length=1,
    )

    topic_axis: TopicAxis

    cosine_similarity: float

    quality_score: float | None = None


# ============================================================
# Internal Retriever result
# ============================================================


class RetrieverResult(BaseModel):
    """
    RetrieverService -> RerankerService 사이에서 사용하는
    내부 데이터 계약.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    title: str

    research_idea: str | None = None

    candidate_count: int = Field(
        ge=0,
    )

    retriever_version: str

    embedding_version: str

    vector_dimension: int = Field(
        ge=1,
    )

    candidates: list[RetrievalCandidate]


# ============================================================
# Final reranked evidence
# ============================================================


class RetrievalEvidence(BaseModel):
    """
    Reranker까지 통과한 최종 Evidence.

    cosine_similarity:
        Retriever 원래 semantic similarity.

    rerank_score:
        Reranker가 여러 신호를 결합해 계산한 최종 점수.

    두 값을 절대로 같은 의미로 사용하지 않는다.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    rank: int = Field(
        ge=1,
    )

    retriever_rank: int = Field(
        ge=1,
    )

    chunk_id: int = Field(
        ge=1,
    )

    document_id: int = Field(
        ge=1,
    )

    chunk_index: int = Field(
        ge=0,
    )

    section_index: int | None = Field(
        default=None,
        ge=0,
    )

    section_heading: str | None = None

    chunk_text: str = Field(
        min_length=1,
    )

    document_title: str = Field(
        min_length=1,
    )

    source: str = Field(
        min_length=1,
    )

    source_id: str = Field(
        min_length=1,
    )

    topic_axis: TopicAxis

    # --------------------------------------------------------
    # Retriever signal
    # --------------------------------------------------------

    cosine_similarity: float

    quality_score: float | None = None

    # --------------------------------------------------------
    # Transparent Reranker signals
    # --------------------------------------------------------

    semantic_score: float = Field(
        ge=0.0,
        le=1.0,
    )

    lexical_score: float = Field(
        ge=0.0,
        le=1.0,
    )

    title_overlap_score: float = Field(
        ge=0.0,
        le=1.0,
    )

    diversity_penalty: float = Field(
        ge=0.0,
    )

    rerank_score: float


# ============================================================
# API Response
# ============================================================


class RetrievalResponse(BaseModel):
    """
    Frontend / Transformer가 공통으로 사용하는
    최종 /retrieve 응답.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    title: str

    research_idea: str | None = None

    candidate_count: int = Field(
        ge=0,
    )

    top_k: int = Field(
        ge=1,
    )

    retriever_version: str

    reranker_version: str

    embedding_version: str

    vector_dimension: int = Field(
        ge=1,
    )

    evidence: list[RetrievalEvidence]