from __future__ import annotations

import logging

from fastapi import (
    APIRouter,
    HTTPException,
    status,
)

from app.schemas.draft import (
    DraftRequest,
    DraftResponse,
)
from app.schemas.retrieval import (
    RetrievalRequest,
)
from app.services.generator_service import (
    get_generator,
)

from app.services.validation_service import (
    validate_draft,
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
    prefix="/draft",
    tags=["Draft"],
)


@router.post(
    "",
    response_model=DraftResponse,
    status_code=status.HTTP_200_OK,
)
def create_draft(
    request: DraftRequest,
) -> DraftResponse:
    """
    TEAM B Draft Pipeline - Phase 2

    Human Input
        ↓
    Retriever
        ↓
    Candidate Top-N
        ↓
    Reranker
        ↓
    Final Evidence Top-K
        ↓
    MockGenerator
        ↓
    Structured Paper JSON

    Validator는 아직 연결하지 않는다.
    """

    try:
        # ====================================================
        # 1. DraftRequest -> RetrievalRequest
        # ====================================================

        retrieval_request = (
            RetrievalRequest(
                title=request.title,

                research_idea=(
                    request.research_idea
                ),

                top_k=(
                    request.top_k
                ),
            )
        )


        # ====================================================
        # 2. Retriever
        # ====================================================

        retriever_result = (
            retrieve_candidates(
                retrieval_request
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


        # ====================================================
        # 3. Reranker
        # ====================================================

        retrieval_response = (
            rerank_candidates(
                request=(
                    retrieval_request
                ),

                retriever_result=(
                    retriever_result
                ),
            )
        )

        if (
            not retrieval_response
            .evidence
        ):
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


        # ====================================================
        # 4. Generator
        # ====================================================

        generator = (
            get_generator()
        )

        draft = generator.generate(
            request=request,
            retrieval=retrieval_response,
        )

        validation = validate_draft(
            sections=draft.sections,
            evidence=draft.evidence,
        )

        draft.validation = validation

        return draft


    # ========================================================
    # Retriever timeout
    # ========================================================

    except RetrieverTimeoutError as exc:
        logger.exception(
            "Draft pipeline retrieval timed out."
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


    # ========================================================
    # Retriever failure
    # ========================================================

    except RetrieverServiceError as exc:
        logger.exception(
            "Draft pipeline retrieval failed."
        )

        raise HTTPException(
            status_code=(
                status
                .HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "Retrieval service is unavailable."
            ),
        ) from exc


    # ========================================================
    # Existing HTTP errors
    # ========================================================

    except HTTPException:
        raise


    # ========================================================
    # Unexpected
    # ========================================================

    except Exception as exc:
        logger.exception(
            "Unexpected draft pipeline error."
        )

        raise HTTPException(
            status_code=(
                status
                .HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Unexpected draft pipeline error."
            ),
        ) from exc