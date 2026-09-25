import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import get_settings
from app.exceptions import DocuMindError
from app.models import ErrorResponse

settings = get_settings()
logging.basicConfig(level=settings.log_level)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(
    title=settings.app_name,
    description="Hybrid-search RAG over your documents, with a real evaluation harness.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(DocuMindError)
async def documind_error_handler(request: Request, exc: DocuMindError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error_code=exc.error_code, detail=str(exc)).model_dump(),
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    first_error = exc.errors()[0]
    detail = f"{'.'.join(str(x) for x in first_error['loc'])}: {first_error['msg']}"
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(error_code="validation_error", detail=detail).model_dump(),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # Anything that reaches here is a bug or an unanticipated failure (e.g. the
    # embedding model's disk cache got corrupted, out of memory, etc.) that
    # none of the named exceptions above caught. Without this handler,
    # FastAPI/Starlette's default response for an unhandled exception is
    # plain text, not JSON -- which breaks any client (like the UI in
    # static/index.html) that expects a JSON error body, and shows up to the
    # end user as a confusing "can't reach the server" even though the
    # server responded just fine. This guarantees every error path returns
    # the same predictable JSON shape, while the real traceback still prints
    # to the server's own terminal/logs for debugging.
    logging.exception("Unhandled exception while processing %s %s", request.method, request.url)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error_code="internal_error",
            detail="Something went wrong on the server. Check the terminal running the app for the full error.",
        ).model_dump(),
    )


app.include_router(router)


@app.get("/")
def root():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    # Fall back to a plain JSON response if the UI file is missing, rather
    # than a confusing 404 -- the API itself still works either way.
    return {"service": settings.app_name, "docs": "/docs"}
