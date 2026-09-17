from fastapi import FastAPI
from fastapi.middleware.cors import (
    CORSMiddleware,
)

from app.api.chunks import (
    router as chunks_router,
)
from app.api.documents import (
    router as documents_router,
)
from app.api.draft import (
    router as draft_router,
)
from app.api.health import (
    router as health_router,
)
from app.api.retrieve import (
    router as retrieve_router,
)
from app.core.config import settings


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title=settings.app_name,

    version="0.4.0",

    description=(
        "Backend API for the Team B AI Paper Agent. "
        "Current phase: Core Corpus + Core Chunks + "
        "Retriever + Reranker + Mock Draft Generation."
    ),
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=(
        settings.cors_origin_list
    ),

    allow_credentials=True,

    allow_methods=[
        "*"
    ],

    allow_headers=[
        "*"
    ],
)


# ============================================================
# Routers
# ============================================================

app.include_router(
    health_router,
    prefix=settings.api_prefix,
)

app.include_router(
    documents_router,
    prefix=settings.api_prefix,
)

app.include_router(
    chunks_router,
    prefix=settings.api_prefix,
)

app.include_router(
    retrieve_router,
    prefix=settings.api_prefix,
)

app.include_router(
    draft_router,
    prefix=settings.api_prefix,
)


# ============================================================
# Root
# ============================================================

@app.get("/")
def root() -> dict:

    return {
        "name": (
            settings.app_name
        ),

        "environment": (
            settings.app_env
        ),

        "api_prefix": (
            settings.api_prefix
        ),

        "docs": (
            "/docs"
        ),

        "status": (
            "running"
        ),

        "backend_stage": (
            "B5-mock-draft-pipeline"
        ),

        "features": {
            "documents": True,
            "chunks": True,
            "retriever": True,
            "reranker": True,
            "draft": True,
            "mock_generator": True,
            "own_transformer": False,
            "validator": True,
        },
    }