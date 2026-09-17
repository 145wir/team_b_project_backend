from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

from app.schemas.retrieval import (
    RetrievalCandidate,
    RetrievalRequest,
    RetrieverResult,
)


logger = logging.getLogger(
    __name__
)


# ============================================================
# Environment defaults
# ============================================================


DEFAULT_CANDIDATE_K = 20

DEFAULT_TIMEOUT_SECONDS = 90

DEFAULT_RAG_BRIDGE_MODULE = (
    "app.rag_bridge"
)


# ============================================================
# Exceptions
# ============================================================


class RetrieverServiceError(
    RuntimeError
):
    """
    Frozen Retriever 호출 실패.
    """


class RetrieverTimeoutError(
    RetrieverServiceError
):
    """
    Frozen Retriever timeout.
    """


# ============================================================
# Environment helpers
# ============================================================


def _get_required_env(
    name: str,
) -> str:
    value = os.getenv(
        name,
        "",
    ).strip()

    if not value:
        raise RetrieverServiceError(
            f"Required environment variable "
            f"{name} is not configured."
        )

    return value


def _get_rag_project_root() -> Path:
    value = _get_required_env(
        "RAG_PROJECT_ROOT"
    )

    root = (
        Path(value)
        .expanduser()
        .resolve()
    )

    if not root.exists():
        raise RetrieverServiceError(
            "RAG project root does not exist."
        )

    if not root.is_dir():
        raise RetrieverServiceError(
            "RAG project root is not a directory."
        )

    bridge_file = (
        root
        / "app"
        / "rag_bridge.py"
    )

    retriever_file = (
        root
        / "app"
        / "rag_retriever.py"
    )

    if not bridge_file.is_file():
        raise RetrieverServiceError(
            "app/rag_bridge.py was not found "
            "in the RAG project."
        )

    if not retriever_file.is_file():
        raise RetrieverServiceError(
            "app/rag_retriever.py was not found "
            "in the RAG project."
        )

    return root


def _get_rag_python() -> Path:
    value = _get_required_env(
        "RAG_PYTHON_EXECUTABLE"
    )

    python_executable = (
        Path(value)
        .expanduser()
        .resolve()
    )

    if not python_executable.is_file():
        raise RetrieverServiceError(
            "RAG Python executable "
            "does not exist."
        )

    return python_executable


def _get_candidate_k(
    requested_top_k: int,
) -> int:
    raw = os.getenv(
        "RETRIEVAL_CANDIDATE_K",
        str(DEFAULT_CANDIDATE_K),
    )

    try:
        candidate_k = int(
            raw
        )

    except ValueError as exc:
        raise RetrieverServiceError(
            "RETRIEVAL_CANDIDATE_K "
            "must be an integer."
        ) from exc

    # 최종 Top-K보다 Candidate 수가 적으면 안 된다.
    candidate_k = max(
        candidate_k,
        requested_top_k,
    )

    return candidate_k


def _get_timeout_seconds() -> int:
    raw = os.getenv(
        "RAG_BRIDGE_TIMEOUT_SECONDS",
        str(DEFAULT_TIMEOUT_SECONDS),
    )

    try:
        timeout = int(
            raw
        )

    except ValueError as exc:
        raise RetrieverServiceError(
            "RAG_BRIDGE_TIMEOUT_SECONDS "
            "must be an integer."
        ) from exc

    if timeout < 1:
        raise RetrieverServiceError(
            "RAG bridge timeout must "
            "be greater than zero."
        )

    return timeout


# ============================================================
# Bridge response normalization
# ============================================================


def _normalize_candidate(
    row: dict[str, Any],
) -> RetrievalCandidate:
    """
    rag_bridge evidence를
    Backend 내부 Candidate 형태로 변환한다.
    """

    return RetrievalCandidate(
        retriever_rank=int(
            row["rank"]
        ),

        chunk_id=int(
            row["chunk_id"]
        ),

        document_id=int(
            row["document_id"]
        ),

        chunk_index=int(
            row.get(
                "chunk_index",
                0,
            )
        ),

        section_index=(
            int(
                row["section_index"]
            )
            if row.get(
                "section_index"
            )
            is not None
            else None
        ),

        section_heading=(
            str(
                row.get(
                    "section_heading",
                    "",
                )
            ).strip()
            or None
        ),

        chunk_text=str(
            row.get(
                "chunk_text",
                "",
            )
        ).strip(),

        document_title=str(
            row.get(
                "document_title",
                row.get(
                    "title",
                    "",
                ),
            )
        ).strip(),

        source=str(
            row.get(
                "source",
                "",
            )
        ).strip(),

        source_id=str(
            row.get(
                "source_id",
                "",
            )
        ).strip(),

        topic_axis=str(
            row.get(
                "topic_axis",
                "",
            )
        ).strip(),

        cosine_similarity=float(
            row["cosine_similarity"]
        ),

        quality_score=(
            float(
                row["quality_score"]
            )
            if row.get(
                "quality_score"
            )
            is not None
            else None
        ),
    )


# ============================================================
# Public service
# ============================================================


def retrieve_candidates(
    request: RetrievalRequest,
) -> RetrieverResult:
    """
    Frozen Vector Retriever를 호출해
    Candidate Top-N을 가져온다.

    이 함수에서는 절대로:

        TfidfVectorizer.fit()
        TruncatedSVD.fit()
        pgvector rebuild
        quality gate 재구현
        reranking

    을 하지 않는다.
    """

    rag_root = (
        _get_rag_project_root()
    )

    rag_python = (
        _get_rag_python()
    )

    candidate_k = (
        _get_candidate_k(
            requested_top_k=(
                request.top_k
            )
        )
    )

    timeout_seconds = (
        _get_timeout_seconds()
    )

    bridge_module = os.getenv(
        "RAG_BRIDGE_MODULE",
        DEFAULT_RAG_BRIDGE_MODULE,
    ).strip()

    if not bridge_module:
        bridge_module = (
            DEFAULT_RAG_BRIDGE_MODULE
        )

    request_payload = {
        "title": (
            request.title
        ),

        "research_idea": (
            request.research_idea
            or ""
        ),

        # 중요:
        # 최종 top_k가 아니라
        # reranker에 넘길 Candidate 수를 요청한다.
        "top_k": (
            candidate_k
        ),
    }

    command = [
        str(
            rag_python
        ),

        "-X",
        "utf8",

        "-m",
        bridge_module,
    ]

    process_environment = (
        os.environ.copy()
    )

    process_environment[
        "PYTHONIOENCODING"
    ] = "utf-8"

    try:
        completed = subprocess.run(
            command,

            cwd=str(
                rag_root
            ),

            input=json.dumps(
                request_payload,
                ensure_ascii=False,
            ),

            text=True,
            encoding="utf-8",

            capture_output=True,

            timeout=timeout_seconds,

            check=False,

            env=process_environment,
        )

    except subprocess.TimeoutExpired as exc:
        logger.exception(
            "Frozen Retriever timed out."
        )

        raise RetrieverTimeoutError(
            "Retriever timed out."
        ) from exc

    except OSError as exc:
        logger.exception(
            "Unable to start Frozen Retriever."
        )

        raise RetrieverServiceError(
            "Unable to start Retriever process."
        ) from exc

    if completed.returncode != 0:
        logger.error(
            (
                "Frozen Retriever failed. "
                "returncode=%s\n"
                "stderr=%s"
            ),
            completed.returncode,
            completed.stderr[
                :5000
            ],
        )

        raise RetrieverServiceError(
            "Frozen Retriever failed."
        )

    raw_stdout = (
        completed.stdout
        .strip()
    )

    if not raw_stdout:
        logger.error(
            (
                "Frozen Retriever returned "
                "empty stdout.\n"
                "stderr=%s"
            ),
            completed.stderr[
                :5000
            ],
        )

        raise RetrieverServiceError(
            "Retriever returned "
            "an empty response."
        )

    try:
        payload = json.loads(
            raw_stdout
        )

    except json.JSONDecodeError as exc:
        logger.error(
            (
                "Frozen Retriever returned "
                "invalid JSON.\n"
                "stdout=%s\n"
                "stderr=%s"
            ),
            raw_stdout[
                :5000
            ],
            completed.stderr[
                :5000
            ],
        )

        raise RetrieverServiceError(
            "Retriever returned "
            "invalid JSON."
        ) from exc

    evidence_rows = (
        payload.get(
            "evidence",
            []
        )
    )

    if not isinstance(
        evidence_rows,
        list,
    ):
        raise RetrieverServiceError(
            "Retriever response field "
            "'evidence' must be a list."
        )

    candidates = [
        _normalize_candidate(
            row
        )
        for row
        in evidence_rows
    ]

    return RetrieverResult(
        title=(
            request.title
        ),

        research_idea=(
            request.research_idea
        ),

        candidate_count=(
            len(
                candidates
            )
        ),

        retriever_version=str(
            payload.get(
                "retriever_version",
                "unknown",
            )
        ),

        embedding_version=str(
            payload.get(
                "embedding_version",
                "unknown",
            )
        ),

        vector_dimension=int(
            payload.get(
                "vector_dimension",
                256,
            )
        ),

        candidates=candidates,
    )