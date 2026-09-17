from __future__ import annotations

from app.schemas.retrieval import (
    RetrievalCandidate,
    RetrievalRequest,
    RetrieverResult,
)
from app.services.reranker_service import (
    RERANKER_VERSION,
    rerank_candidates,
)


# ============================================================
# Helpers
# ============================================================


def make_candidate(
    *,
    retriever_rank: int,
    chunk_id: int,
    document_id: int,
    chunk_text: str,
    document_title: str,
    cosine_similarity: float,
    quality_score: float = 0.90,
    topic_axis: str = "onboard_ai",
) -> RetrievalCandidate:
    """
    테스트 Candidate 생성 helper.
    """

    return RetrievalCandidate(
        retriever_rank=retriever_rank,

        chunk_id=chunk_id,

        document_id=document_id,

        chunk_index=0,

        section_index=1,

        section_heading="Introduction",

        chunk_text=chunk_text,

        document_title=document_title,

        source="ntrs",

        source_id=(
            f"TEST-{chunk_id}"
        ),

        topic_axis=topic_axis,

        cosine_similarity=(
            cosine_similarity
        ),

        quality_score=(
            quality_score
        ),
    )


def make_request(
    top_k: int = 2,
) -> RetrievalRequest:
    return RetrievalRequest(
        title=(
            "Autonomous Onboard Decision-Making "
            "for Deep-Space Spacecraft"
        ),

        research_idea=(
            "심우주 통신 지연 상황에서 "
            "온보드 AI의 자율 의사결정을 연구한다."
        ),

        top_k=top_k,
    )


def make_retriever_result(
    request: RetrievalRequest,
    candidates: list[
        RetrievalCandidate
    ],
) -> RetrieverResult:
    return RetrieverResult(
        title=request.title,

        research_idea=(
            request.research_idea
        ),

        candidate_count=(
            len(
                candidates
            )
        ),

        retriever_version=(
            "test_retriever_v1"
        ),

        embedding_version=(
            "test_svd256"
        ),

        vector_dimension=256,

        candidates=candidates,
    )


# ============================================================
# Test 1
# Basic Top-K
# ============================================================


def test_reranker_returns_requested_top_k() -> None:
    """
    Candidate 3개를 넣고 top_k=2 요청 시
    최종 Evidence가 정확히 2개인지 검사한다.
    """

    request = make_request(
        top_k=2
    )

    candidates = [
        make_candidate(
            retriever_rank=1,
            chunk_id=1,
            document_id=101,
            chunk_text=(
                "Deep-space spacecraft require "
                "autonomous onboard decision-making "
                "when communication delays prevent "
                "real-time ground intervention."
            ),
            document_title=(
                "Autonomous Decision Making "
                "for Deep Space Missions"
            ),
            cosine_similarity=0.72,
        ),

        make_candidate(
            retriever_rank=2,
            chunk_id=2,
            document_id=102,
            chunk_text=(
                "Onboard artificial intelligence "
                "supports spacecraft autonomy under "
                "long communication latency."
            ),
            document_title=(
                "Onboard AI for "
                "Autonomous Spacecraft"
            ),
            cosine_similarity=0.68,
        ),

        make_candidate(
            retriever_rank=3,
            chunk_id=3,
            document_id=103,
            chunk_text=(
                "Planetary rover wheel mechanics "
                "and soil interaction are discussed."
            ),
            document_title=(
                "Rover Wheel Mechanics"
            ),
            cosine_similarity=0.40,
            topic_axis="rover_autonomy",
        ),
    ]

    retriever_result = (
        make_retriever_result(
            request,
            candidates,
        )
    )

    result = rerank_candidates(
        request=request,
        retriever_result=(
            retriever_result
        ),
    )

    assert result.top_k == 2

    assert len(
        result.evidence
    ) == 2

    assert (
        result.candidate_count
        == 3
    )

    assert (
        result.reranker_version
        == RERANKER_VERSION
    )

    assert (
        result.vector_dimension
        == 256
    )

    # 최종 rank는 1부터 연속이어야 한다.
    assert [
        item.rank
        for item
        in result.evidence
    ] == [
        1,
        2,
    ]

    # 모든 Evidence에 reranking signal이 있어야 한다.
    for item in result.evidence:
        assert (
            item.rerank_score
            is not None
        )

        assert (
            0.0
            <= item.semantic_score
            <= 1.0
        )

        assert (
            0.0
            <= item.lexical_score
            <= 1.0
        )

        assert (
            0.0
            <= item.title_overlap_score
            <= 1.0
        )


# ============================================================
# Test 2
# Reranking really changes ranking
# ============================================================


def test_reranker_can_promote_more_relevant_candidate() -> None:
    """
    Retriever cosine만 보면 Candidate 1이 1위지만,
    Candidate 2가 query/title과 훨씬 관련성이 높으면
    Reranker가 Candidate 2를 최종 1위로 올릴 수 있는지 검사.

    즉:
        Retriever Rank != Final Rank

    가 실제로 발생할 수 있어야 한다.
    """

    request = make_request(
        top_k=2
    )

    candidates = [
        # ----------------------------------------------------
        # Retriever에서는 1등이지만
        # query 내용과는 거의 관계없게 구성.
        # ----------------------------------------------------
        make_candidate(
            retriever_rank=1,
            chunk_id=10,
            document_id=201,
            chunk_text=(
                "Electrical power systems, battery "
                "temperature regulation, and solar "
                "array performance are evaluated."
            ),
            document_title=(
                "Spacecraft Electrical "
                "Power System Analysis"
            ),
            cosine_similarity=0.700,
            quality_score=0.90,
        ),

        # ----------------------------------------------------
        # cosine은 아주 조금 낮지만
        # Title/Idea와 직접적으로 관련됨.
        # ----------------------------------------------------
        make_candidate(
            retriever_rank=2,
            chunk_id=20,
            document_id=202,
            chunk_text=(
                "Autonomous onboard decision-making "
                "allows deep-space spacecraft to "
                "respond locally when communication "
                "delay prevents real-time ground control."
            ),
            document_title=(
                "Autonomous Onboard Decision-Making "
                "for Deep-Space Spacecraft"
            ),
            cosine_similarity=0.690,
            quality_score=0.95,
        ),

        # semantic min/max normalization용
        # 낮은 관련 후보
        make_candidate(
            retriever_rank=3,
            chunk_id=30,
            document_id=203,
            chunk_text=(
                "Mechanical properties of rover "
                "wheel materials are measured."
            ),
            document_title=(
                "Mechanical Rover Wheel Design"
            ),
            cosine_similarity=0.100,
            quality_score=0.80,
            topic_axis="rover_autonomy",
        ),
    ]

    result = rerank_candidates(
        request=request,

        retriever_result=(
            make_retriever_result(
                request,
                candidates,
            )
        ),
    )

    first = result.evidence[0]

    # Retriever에서는 2위였던 Candidate가
    # 최종 Reranker에서는 1위가 되어야 한다.
    assert first.chunk_id == 20

    assert (
        first.retriever_rank
        == 2
    )

    assert (
        first.rank
        == 1
    )

    assert (
        first.rerank_score
        >
        result.evidence[1]
        .rerank_score
    )

    # Query와 완전히 일치하는 title이므로
    # title overlap도 높아야 한다.
    assert (
        first.title_overlap_score
        > 0.5
    )


# ============================================================
# Test 3
# Document diversity
# ============================================================


def test_reranker_limits_duplicate_document_chunks() -> None:
    """
    같은 document에서 매우 높은 점수의 chunk가
    여러 개 있더라도 충분한 다른 문서가 존재하면
    하나의 document가 최종 Evidence를 독점하지
    않는지 검사한다.

    현재 Reranker V1:
        MAX_CHUNKS_PER_DOCUMENT = 2
    """

    request = make_request(
        top_k=3
    )

    candidates = [
        # 같은 document 300
        make_candidate(
            retriever_rank=1,
            chunk_id=101,
            document_id=300,
            chunk_text=(
                "Autonomous onboard decision-making "
                "for deep-space spacecraft."
            ),
            document_title=(
                "Deep Space Autonomous Systems"
            ),
            cosine_similarity=0.90,
        ),

        make_candidate(
            retriever_rank=2,
            chunk_id=102,
            document_id=300,
            chunk_text=(
                "Communication delay motivates "
                "spacecraft onboard autonomy."
            ),
            document_title=(
                "Deep Space Autonomous Systems"
            ),
            cosine_similarity=0.88,
        ),

        make_candidate(
            retriever_rank=3,
            chunk_id=103,
            document_id=300,
            chunk_text=(
                "Autonomous planning reduces "
                "dependence on ground control."
            ),
            document_title=(
                "Deep Space Autonomous Systems"
            ),
            cosine_similarity=0.86,
        ),

        # 다른 document
        make_candidate(
            retriever_rank=4,
            chunk_id=201,
            document_id=400,
            chunk_text=(
                "Onboard AI supports autonomous "
                "spacecraft operation under "
                "communication latency."
            ),
            document_title=(
                "Onboard AI and "
                "Spacecraft Autonomy"
            ),
            cosine_similarity=0.75,
        ),
    ]

    result = rerank_candidates(
        request=request,

        retriever_result=(
            make_retriever_result(
                request,
                candidates,
            )
        ),
    )

    assert len(
        result.evidence
    ) == 3

    document_300_count = sum(
        item.document_id == 300

        for item
        in result.evidence
    )

    # 동일 논문은 최대 2개까지만 선택
    assert (
        document_300_count
        <= 2
    )

    # 다른 document도 최종 결과에 들어와야 함
    selected_document_ids = {
        item.document_id

        for item
        in result.evidence
    }

    assert 400 in (
        selected_document_ids
    )