from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.core.config import settings


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description=(
        "Backend API for the Team B AI Paper Agent. "
        "Current phase: AWS RDS Core Corpus access."
    ),
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(
    health_router,
    prefix=settings.api_prefix,
)

app.include_router(
    documents_router,
    prefix=settings.api_prefix,
)


@app.get("/")
def root() -> dict:
    return {
        "name": settings.app_name,
        "environment": settings.app_env,
        "api_prefix": settings.api_prefix,
        "docs": "/docs",
        "status": "running",
    }