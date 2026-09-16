from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from app.repositories.chunk_repository import (
    count_chunks,
    list_chunks,
)
from app.repositories.document_repository import (
    count_documents,
    get_document_by_id,
    get_document_stats,
    list_documents,
)
from app.schemas.chunk import (
    DocumentChunksResponse,
)
from app.schemas.document import (
    DocumentDetail,
    DocumentListResponse,
    DocumentStatsResponse,
)


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


Source = Literal[
    "arxiv",
    "ntrs",
]

TopicAxis = Literal[
    "rover_autonomy",
    "onboard_ai",
    "satellite_autonomy",
]


@router.get(
    "",
    response_model=DocumentListResponse,
)
def read_documents(
    source: Source | None = Query(
        default=None,
        description=(
            "Filter by source: "
            "arxiv or ntrs"
        ),
    ),
    topic_axis: TopicAxis | None = Query(
        default=None,
        description=(
            "Filter by Team B "
            "research topic axis"
        ),
    ),
    parse_status: str | None = Query(
        default=None,
        description=(
            "Filter by parse/quality status"
        ),
    ),
    limit: int = Query(
        default=50,
        ge=1,
        le=200,
    ),
    offset: int = Query(
        default=0,
        ge=0,
    ),
) -> DocumentListResponse:
    """
    Core Corpus document 목록.

    raw_content / clean_content는
    목록 응답에서 제외한다.
    """

    documents = list_documents(
        source=source,
        topic_axis=topic_axis,
        parse_status=parse_status,
        limit=limit,
        offset=offset,
    )

    total = count_documents(
        source=source,
        topic_axis=topic_axis,
        parse_status=parse_status,
    )

    return DocumentListResponse(
        total=total,
        limit=limit,
        offset=offset,
        items=documents,
    )


@router.get(
    "/stats",
    response_model=DocumentStatsResponse,
)
def read_document_stats() -> DocumentStatsResponse:
    """
    Core Corpus 문서 통계.
    """

    stats = get_document_stats()

    return DocumentStatsResponse(
        **stats
    )


@router.get(
    "/{document_id}/chunks",
    response_model=DocumentChunksResponse,
)
def read_document_chunks(
    document_id: int,
    limit: int = Query(
        default=100,
        ge=1,
        le=500,
    ),
    offset: int = Query(
        default=0,
        ge=0,
    ),
) -> DocumentChunksResponse:
    """
    특정 Core Document에 속한 Chunk 목록.

    RAG 디버깅에서 매우 유용하다.
    """

    document = get_document_by_id(
        document_id
    )

    if document is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Document {document_id} "
                "was not found."
            ),
        )

    chunks = list_chunks(
        document_id=document_id,
        limit=limit,
        offset=offset,
    )

    total = count_chunks(
        document_id=document_id,
    )

    return DocumentChunksResponse(
        document_id=document_id,
        total=total,
        limit=limit,
        offset=offset,
        items=chunks,
    )


@router.get(
    "/{document_id}",
    response_model=DocumentDetail,
)
def read_document(
    document_id: int,
) -> DocumentDetail:
    """
    Core document 상세 조회.

    raw_content와 clean_content까지 포함한다.
    """

    document = get_document_by_id(
        document_id
    )

    if document is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Document {document_id} "
                "was not found."
            ),
        )

    return DocumentDetail(
        **document
    )