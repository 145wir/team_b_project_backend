from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.core.database import check_database_connection


router = APIRouter(
    prefix="/health",
    tags=["Health"],
)


@router.get("")
def health_check() -> dict:
    """
    Check backend and database connectivity.
    """

    try:
        database_info = check_database_connection()

        return {
            "status": "ok",
            "application": settings.app_name,
            "environment": settings.app_env,
            "database": {
                "status": "connected",
                "name": database_info["database_name"],
                "schema": database_info["schema_name"],
            },
        }

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Database connection failed: {exc}",
        ) from exc