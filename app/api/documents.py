from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from app.repositories.document_repository import (
    count_documents,
    get_document_by_id,
    get_document_stats,
    list_documents,
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
    source: str | None = Query(
        default=None,
        description="Filter by source, e.g. arxiv or ntrs",
    ),
    topic_axis: TopicAxis | None = Query(
        default=None,
        description="Filter by Team B research topic axis",
    ),
    parse_status: str | None = Query(
        default=None,
        description="Filter by parser/quality status",
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
    Return Core Corpus documents.

    Full raw/clean content is not included here.
    Use GET /documents/{id} for full text.
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
    Return Core Corpus statistics.
    """

    stats = get_document_stats()

    return DocumentStatsResponse(**stats)


@router.get(
    "/{document_id}",
    response_model=DocumentDetail,
)
def read_document(
    document_id: int,
) -> DocumentDetail:
    """
    Return one Core document including full raw/clean content.
    """

    document = get_document_by_id(document_id)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail=f"Document {document_id} was not found.",
        )

    return DocumentDetail(**document)