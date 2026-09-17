from __future__ import annotations

import logging

from fastapi import (
    APIRouter,
    HTTPException,
    status,
)

from app.schemas.retrieval import (
    RetrievalRequest,
    RetrievalResponse,
)
from app.services.reranker_service import (
    rerank_candidates,
)
from app.services.retriever_service import (
    RetrieverServiceError,
    RetrieverTimeoutError,
    retrieve_candidates,
)


logger = logging.getLogger(
    __name__
)


router = APIRouter(
    prefix="/retrieve",
    tags=["Retrieval"],
)


@router.post(
    "",
    response_model=RetrievalResponse,
    status_code=status.HTTP_200_OK,
)
def retrieve(
    request: RetrievalRequest,
) -> RetrievalResponse:
    """
    TEAM B Retrieval Pipeline

    Human Title
        +
    Human Research Idea
        ↓
    Vector Retriever
        ↓
    Candidate Top-N
        ↓
    Reranker
        ↓
    Final Top-K Evidence
    """

    try:
        # ----------------------------------------------------
        # 1. Vector Retriever
        # ----------------------------------------------------

        retriever_result = (
            retrieve_candidates(
                request
            )
        )

        if (
            not retriever_result
            .candidates
        ):
            raise HTTPException(
                status_code=(
                    status
                    .HTTP_404_NOT_FOUND
                ),
                detail=(
                    "No retrieval candidates "
                    "were found."
                ),
            )

        # ----------------------------------------------------
        # 2. Reranker
        # ----------------------------------------------------

        response = (
            rerank_candidates(
                request=request,
                retriever_result=(
                    retriever_result
                ),
            )
        )

        if not response.evidence:
            raise HTTPException(
                status_code=(
                    status
                    .HTTP_404_NOT_FOUND
                ),
                detail=(
                    "No evidence remained "
                    "after reranking."
                ),
            )

        return response

    except RetrieverTimeoutError as exc:
        logger.exception(
            "Retriever timed out."
        )

        raise HTTPException(
            status_code=(
                status
                .HTTP_504_GATEWAY_TIMEOUT
            ),
            detail=(
                "Retrieval service timed out."
            ),
        ) from exc

    except RetrieverServiceError as exc:
        logger.exception(
            "Retriever service failed."
        )

        raise HTTPException(
            status_code=(
                status
                .HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "Retrieval service "
                "is unavailable."
            ),
        ) from exc

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "Unexpected retrieval pipeline error."
        )

        raise HTTPException(
            status_code=(
                status
                .HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Unexpected retrieval "
                "pipeline error."
            ),
        ) from exc