from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from app.schemas.retrieval import (
    RetrievalCandidate,
    RetrievalEvidence,
    RetrievalRequest,
    RetrievalResponse,
    RetrieverResult,
)


# ============================================================
# Version
# ============================================================


RERANKER_VERSION = (
    "transparent_lightweight_reranker_v1"
)


# ============================================================
# Baseline weights
# ============================================================
#
# 중요:
# 이 값들은 아직 "최적값"이라고 주장하지 않는다.
#
# Retriever Only
# vs
# Retriever + Reranker
#
# evaluation을 통해 나중에 조정한다.
# ============================================================


SEMANTIC_WEIGHT = 0.55

LEXICAL_WEIGHT = 0.20

TITLE_WEIGHT = 0.15

QUALITY_WEIGHT = 0.10


# 같은 document에서 Evidence가 과도하게 나오는 것을
# 방지하기 위한 간단한 diversity 설정.
MAX_CHUNKS_PER_DOCUMENT = 2

DUPLICATE_DOCUMENT_PENALTY = 0.12


# ============================================================
# Stopwords
# ============================================================


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "using",
    "via",
    "was",
    "were",
    "with",
}


# ============================================================
# Internal scored candidate
# ============================================================


@dataclass
class ScoredCandidate:
    candidate: RetrievalCandidate

    semantic_score: float

    lexical_score: float

    title_overlap_score: float

    quality_component: float

    base_score: float

    diversity_penalty: float = 0.0

    final_score: float = 0.0


# ============================================================
# Token helpers
# ============================================================


TOKEN_PATTERN = re.compile(
    r"[A-Za-z0-9가-힣]+(?:[-_][A-Za-z0-9가-힣]+)*"
)


def _tokenize(
    text: str,
) -> set[str]:
    """
    Lightweight lexical tokenization.

    Reranker V1은 외부 pretrained 모델을 사용하지 않는다.
    """

    tokens = {
        token.lower()

        for token
        in TOKEN_PATTERN.findall(
            text or ""
        )

        if token
    }

    return {
        token

        for token
        in tokens

        if token
        not in STOPWORDS

        and len(
            token
        )
        >= 2
    }


def _build_query_text(
    request: RetrievalRequest,
) -> str:
    if request.research_idea:
        return (
            f"{request.title}\n"
            f"{request.research_idea}"
        )

    return request.title


# ============================================================
# Score helpers
# ============================================================


def _min_max_normalize(
    values: list[float],
) -> list[float]:
    if not values:
        return []

    minimum = min(
        values
    )

    maximum = max(
        values
    )

    if math.isclose(
        minimum,
        maximum,
    ):
        # 모든 후보의 cosine이 같다면
        # semantic component에서 차이를 만들지 않는다.
        return [
            1.0
            for _ in values
        ]

    scale = (
        maximum
        - minimum
    )

    return [
        (
            value
            - minimum
        )
        / scale

        for value
        in values
    ]


def _jaccard_score(
    query_tokens: set[str],
    text_tokens: set[str],
) -> float:
    if (
        not query_tokens
        or not text_tokens
    ):
        return 0.0

    intersection = (
        query_tokens
        & text_tokens
    )

    union = (
        query_tokens
        | text_tokens
    )

    if not union:
        return 0.0

    return (
        len(
            intersection
        )
        / len(
            union
        )
    )


def _query_coverage_score(
    query_tokens: set[str],
    target_tokens: set[str],
) -> float:
    """
    Query 핵심 token 중 몇 %가
    target(title)에 포함되는지 측정.
    """

    if not query_tokens:
        return 0.0

    overlap = (
        query_tokens
        & target_tokens
    )

    return (
        len(
            overlap
        )
        / len(
            query_tokens
        )
    )


def _quality_component(
    candidate: RetrievalCandidate,
) -> float:
    """
    Quality Gate가 이미 계산한 값이 있으면 사용.

    값이 없으면 중립값 0.5를 사용한다.
    """

    if candidate.quality_score is None:
        return 0.5

    return max(
        0.0,
        min(
            1.0,
            float(
                candidate.quality_score
            ),
        ),
    )


# ============================================================
# Candidate scoring
# ============================================================


def _score_candidates(
    request: RetrievalRequest,
    candidates: list[
        RetrievalCandidate
    ],
) -> list[
    ScoredCandidate
]:
    if not candidates:
        return []

    query_text = _build_query_text(
        request
    )

    query_tokens = _tokenize(
        query_text
    )

    user_title_tokens = _tokenize(
        request.title
    )

    cosine_values = [
        candidate.cosine_similarity

        for candidate
        in candidates
    ]

    semantic_scores = (
        _min_max_normalize(
            cosine_values
        )
    )

    scored: list[
        ScoredCandidate
    ] = []

    for (
        candidate,
        semantic_score,
    ) in zip(
        candidates,
        semantic_scores,
        strict=True,
    ):
        chunk_tokens = (
            _tokenize(
                candidate.chunk_text
            )
        )

        title_tokens = (
            _tokenize(
                candidate.document_title
            )
        )

        lexical_score = (
            _jaccard_score(
                query_tokens,
                chunk_tokens,
            )
        )

        title_score = _query_coverage_score(
            user_title_tokens,
            title_tokens,
        )

        quality_score = (
            _quality_component(
                candidate
            )
        )

        base_score = (
            semantic_score
            * SEMANTIC_WEIGHT

            + lexical_score
            * LEXICAL_WEIGHT

            + title_score
            * TITLE_WEIGHT

            + quality_score
            * QUALITY_WEIGHT
        )

        scored.append(
            ScoredCandidate(
                candidate=candidate,

                semantic_score=(
                    semantic_score
                ),

                lexical_score=(
                    lexical_score
                ),

                title_overlap_score=(
                    title_score
                ),

                quality_component=(
                    quality_score
                ),

                base_score=(
                    base_score
                ),

                final_score=(
                    base_score
                ),
            )
        )

    scored.sort(
        key=lambda item: (
            item.base_score,
            item.candidate.cosine_similarity,
        ),
        reverse=True,
    )

    return scored


# ============================================================
# Diversity-aware selection
# ============================================================


def _select_with_diversity(
    scored_candidates: list[
        ScoredCandidate
    ],
    top_k: int,
) -> list[
    ScoredCandidate
]:
    """
    Greedy diversity selection.

    동일 document가 지나치게 Evidence를 독점하지 않도록 한다.

    1차:
        document당 최대 MAX_CHUNKS_PER_DOCUMENT

    부족하면:
        남은 후보에서 다시 채운다.
    """

    selected: list[
        ScoredCandidate
    ] = []

    selected_chunk_ids: set[int] = set()

    document_counts: Counter[int] = (
        Counter()
    )

    # --------------------------------------------------------
    # First pass
    # --------------------------------------------------------

    for item in scored_candidates:
        document_id = (
            item.candidate.document_id
        )

        current_count = (
            document_counts[
                document_id
            ]
        )

        if (
            current_count
            >= MAX_CHUNKS_PER_DOCUMENT
        ):
            continue

        penalty = (
            current_count
            * DUPLICATE_DOCUMENT_PENALTY
        )

        item.diversity_penalty = (
            penalty
        )

        item.final_score = (
            item.base_score
            - penalty
        )

        selected.append(
            item
        )

        selected_chunk_ids.add(
            item.candidate.chunk_id
        )

        document_counts[
            document_id
        ] += 1

        if (
            len(
                selected
            )
            >= top_k
        ):
            break

    # --------------------------------------------------------
    # Fallback
    #
    # Candidate가 부족해서 top_k를 채우지 못했을 경우
    # document 제한을 완화한다.
    # --------------------------------------------------------

    if (
        len(
            selected
        )
        < top_k
    ):
        for item in scored_candidates:
            chunk_id = (
                item.candidate.chunk_id
            )

            if (
                chunk_id
                in selected_chunk_ids
            ):
                continue

            document_id = (
                item.candidate.document_id
            )

            current_count = (
                document_counts[
                    document_id
                ]
            )

            penalty = (
                current_count
                * DUPLICATE_DOCUMENT_PENALTY
            )

            item.diversity_penalty = (
                penalty
            )

            item.final_score = (
                item.base_score
                - penalty
            )

            selected.append(
                item
            )

            selected_chunk_ids.add(
                chunk_id
            )

            document_counts[
                document_id
            ] += 1

            if (
                len(
                    selected
                )
                >= top_k
            ):
                break

    # diversity penalty 적용 이후 다시 정렬
    selected.sort(
        key=lambda item: (
            item.final_score,
            item.candidate.cosine_similarity,
        ),
        reverse=True,
    )

    return selected[
        :top_k
    ]


# ============================================================
# Public service
# ============================================================


def rerank_candidates(
    request: RetrievalRequest,
    retriever_result: RetrieverResult,
) -> RetrievalResponse:
    """
    Candidate Top-N
        ↓
    transparent lightweight reranker
        ↓
    Final Top-K Evidence
    """

    scored_candidates = (
        _score_candidates(
            request=request,
            candidates=(
                retriever_result
                .candidates
            ),
        )
    )

    selected = (
        _select_with_diversity(
            scored_candidates=(
                scored_candidates
            ),
            top_k=request.top_k,
        )
    )

    evidence: list[
        RetrievalEvidence
    ] = []

    for (
        final_rank,
        item,
    ) in enumerate(
        selected,
        start=1,
    ):
        candidate = (
            item.candidate
        )

        evidence.append(
            RetrievalEvidence(
                rank=final_rank,

                retriever_rank=(
                    candidate
                    .retriever_rank
                ),

                chunk_id=(
                    candidate
                    .chunk_id
                ),

                document_id=(
                    candidate
                    .document_id
                ),

                chunk_index=(
                    candidate
                    .chunk_index
                ),

                section_index=(
                    candidate
                    .section_index
                ),

                section_heading=(
                    candidate
                    .section_heading
                ),

                chunk_text=(
                    candidate
                    .chunk_text
                ),

                document_title=(
                    candidate
                    .document_title
                ),

                source=(
                    candidate
                    .source
                ),

                source_id=(
                    candidate
                    .source_id
                ),

                topic_axis=(
                    candidate
                    .topic_axis
                ),

                cosine_similarity=(
                    candidate
                    .cosine_similarity
                ),

                quality_score=(
                    candidate
                    .quality_score
                ),

                semantic_score=round(
                    item.semantic_score,
                    6,
                ),

                lexical_score=round(
                    item.lexical_score,
                    6,
                ),

                title_overlap_score=round(
                    item.title_overlap_score,
                    6,
                ),

                diversity_penalty=round(
                    item.diversity_penalty,
                    6,
                ),

                rerank_score=round(
                    item.final_score,
                    6,
                ),
            )
        )

    return RetrievalResponse(
        title=(
            retriever_result.title
        ),

        research_idea=(
            retriever_result
            .research_idea
        ),

        candidate_count=(
            retriever_result
            .candidate_count
        ),

        top_k=(
            len(
                evidence
            )
        ),

        retriever_version=(
            retriever_result
            .retriever_version
        ),

        reranker_version=(
            RERANKER_VERSION
        ),

        embedding_version=(
            retriever_result
            .embedding_version
        ),

        vector_dimension=(
            retriever_result
            .vector_dimension
        ),

        evidence=evidence,
    )