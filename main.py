"""Local-only API: retrieve authored facts first; use the toy model as fallback."""

import logging
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from knowledge_base import KnowledgeBase
from model import TinyNgramModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("case_ai")
ROOT = Path(__file__).parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "artifacts/model.json"))
KNOWLEDGE_BASE_PATH = Path(os.getenv("KNOWLEDGE_BASE_PATH", ROOT / "data/knowledge_base.json"))
MAX_MESSAGE_LENGTH = 2000
model = None
knowledge_base = None
try:
    if MODEL_PATH.is_file():
        model = TinyNgramModel.load(MODEL_PATH)
except Exception:
    logger.exception("Could not load the local model")
try:
    if KNOWLEDGE_BASE_PATH.is_file():
        knowledge_base = KnowledgeBase.load(KNOWLEDGE_BASE_PATH)
except Exception:
    logger.exception("Could not load the local knowledge base")

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
    return {"status": "ok", "model_loaded": model is not None, "knowledge_base_loaded": knowledge_base is not None}


@app.post("/api/chat")
def chat(data: ChatRequest):
    message = data.message.strip()
    if not message:
        return JSONResponse(status_code=422, content={"error": "پیام نمی‌تواند خالی باشد."})

    try:
        if knowledge_base is not None:
            result = knowledge_base.search(message)
            if result:
                return {"response": result["answer"], "source": "knowledge_base"}
        if model is None:
            return JSONResponse(
                status_code=503,
                content={"error": "مدل آموزش‌دیده پیدا نشد؛ ابتدا دستور آموزش نمونه را اجرا کنید."},
            )
        reply = model.generate(message)
    except Exception:
        logger.exception("Chat processing failed")
        return JSONResponse(status_code=500, content={"error": "ساخت پاسخ ممکن نشد."})
    if not reply:
        reply = "در پایگاه دانش پاسخ مناسبی پیدا نشد و مدل آموزشی نیز پاسخ قابل استفاده‌ای نساخت."
    return {"response": reply, "source": "toy_model"}
