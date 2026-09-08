"""ASGI entry point used by Uvicorn to run the MediBot FastAPI application."""

from app.api.application import create_application


# This application object is the production ASGI entry point: uvicorn app.main:app.
app = create_application()
