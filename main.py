"""Local-only FastAPI service for the from-scratch educational model."""

import logging
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from model import TinyNgramModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("case_ai")
MODEL_PATH = Path(os.getenv("MODEL_PATH", Path(__file__).parent / "artifacts/model.json"))
MAX_MESSAGE_LENGTH = 2000
model = None
try:
    if MODEL_PATH.is_file():
        model = TinyNgramModel.load(MODEL_PATH)
except Exception:
    # Do not expose local paths or parsing details to clients.
    logger.exception("Could not load the local model")

app = FastAPI(title="Case educational Persian toy model", docs_url=None, redoc_url=None)
origins = [origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "").split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_request: Request, _exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"error": "درخواست نامعتبر است؛ پیام باید بین ۱ تا ۲۰۰۰ نویسه باشد."})


@app.exception_handler(Exception)
async def unexpected_error_handler(_request: Request, exc: Exception):
    logger.exception("Unexpected request failure", exc_info=exc)
    return JSONResponse(status_code=500, content={"error": "خطای داخلی سرور رخ داد."})


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/api/chat")
def chat(data: ChatRequest):
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"error": "مدل آموزش‌دیده پیدا نشد؛ ابتدا دستور آموزش نمونه را اجرا کنید."},
        )
    message = data.message.strip()
    if not message:
        return JSONResponse(status_code=422, content={"error": "پیام نمی‌تواند خالی باشد."})
    try:
        reply = model.generate(message)
    except Exception:
        logger.exception("Text generation failed")
        return JSONResponse(status_code=500, content={"error": "ساخت پاسخ ممکن نشد."})
    if not reply:
        reply = "این نمونهٔ آموزشی پاسخ مناسبی برای این ورودی پیدا نکرد."
    return {"response": reply}
