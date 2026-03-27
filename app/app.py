import hashlib
import logging
import os
import re
from functools import lru_cache
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from utils import GigaChatClient, LocalEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("gigachat-rag-api")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "knowledge_base")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", 384))
SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", 0.36))
MAX_CONTEXT_HITS = int(os.getenv("MAX_CONTEXT_HITS", 3))
RETURN_PLACEHOLDERS = os.getenv("RETURN_PLACEHOLDERS", "false").lower() in {"1", "true", "yes"}
PLACEHOLDER_TITLE = os.getenv("PLACEHOLDER_TITLE", "Уточняющий вопрос")

app = FastAPI(title="gigachat-rag-vop", version="1.0.0")


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)


class QuestionUpdateRequest(BaseModel):
    question_id: int


class QuestionBulkUpdateRequest(BaseModel):
    question_ids: list[int]


class EmbeddingRequest(BaseModel):
    query: str
    for_search: bool = False


class EmbeddingResponse(BaseModel):
    embedding: list[float]
    size: int


class SearchResultItem(BaseModel):
    question_id: int
    question_text: str
    answer_id: int
    answer_text: str
    answer_name: str
    score: float


@lru_cache(maxsize=1)
def get_neural_models() -> tuple[GigaChatClient, LocalEncoder]:
    return GigaChatClient(), LocalEncoder()


@lru_cache(maxsize=1)
def get_db_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def _normalize_query(text: str) -> str:
    return _normalize_text(text).lower()


def _stable_id(prefix: str, value: str) -> int:
    digest = hashlib.md5(f"{prefix}:{value}".encode("utf-8")).hexdigest()[:8]
    return int(digest, 16)


def _is_context_dependent(query: str) -> bool:
    normalized = _normalize_query(query)
    if not normalized:
        return True

    ambiguous_patterns = [
        r"\b(он|она|оно|они|это|этот|эта|эти|тот|та|те)\b",
        r"\b(там|тогда|такое|такой|такая|так)\b",
        r"\b(а если|и если|а сколько|а какие|а что|а как)\b",
        r"\b(нужно|стоит|делать|получить)\b",
    ]
    short_query = len(normalized.split()) <= 4
    return short_query and any(re.search(pattern, normalized) for pattern in ambiguous_patterns)


def _extract_group_hits(query_vector: list[float], limit: int) -> list[dict[str, Any]]:
    qdrant = get_db_client()
    grouped = qdrant.query_points_groups(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        group_by="answer_hash",
        limit=limit,
        with_payload=True,
        group_size=1,
    )
    groups = getattr(grouped, "groups", grouped)
    hits: list[dict[str, Any]] = []
    for group in groups:
        group_hits = getattr(group, "hits", None) or []
        if not group_hits:
            continue
        hit = group_hits[0]
        payload = hit.payload or {}
        hits.append(
            {
                "score": float(getattr(hit, "score", 0.0)),
                "topic": _normalize_text(str(payload.get("topic", "")).strip()),
                "question": _normalize_text(str(payload.get("question", "")).strip()),
                "answer": _normalize_text(str(payload.get("answer", "")).strip()),
                "answer_hash": str(payload.get("answer_hash", "")),
            }
        )
    return hits


def _search_knowledge(query: str, limit: int) -> list[dict[str, Any]]:
    _, encoder = get_neural_models()
    query_vector = encoder.get_embedding(_normalize_query(query))
    try:
        return _extract_group_hits(query_vector=query_vector, limit=limit)
    except Exception:
        logger.exception("Grouped search in Qdrant failed")
        return []


def _select_context_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered = [hit for hit in hits if hit["score"] >= SCORE_THRESHOLD and hit["answer"]]
    return filtered[:MAX_CONTEXT_HITS]


def _build_context(context_hits: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for idx, hit in enumerate(context_hits, start=1):
        topic = hit["topic"] or "Общая тема"
        question = hit["question"] or "Не указан"
        answer = hit["answer"]
        blocks.append(
            f"Источник {idx}\n"
            f"Тема: {topic}\n"
            f"Похожий вопрос: {question}\n"
            f"Ответ из базы: {answer}"
        )
    return "\n\n".join(blocks)


def _build_system_prompt(question: str, context_hits: list[dict[str, Any]]) -> str:
    has_context = bool(context_hits)
    context_block = _build_context(context_hits) if has_context else "Контекст из базы отсутствует."
    ambiguity_note = (
        "Запрос выглядит зависимым от предыдущего контекста. "
        "Если из текста вопроса нельзя понять, о чем речь, задай один короткий уточняющий вопрос."
        if _is_context_dependent(question)
        else "Если вопрос сформулирован понятно, отвечай сразу."
    )

    return (
        "Ты консультант МФЦ. Отвечай по-русски, понятно и без канцелярита.\n\n"
        "Главные правила:\n"
        "- Используй только факты из контекста базы, если контекст есть.\n"
        "- Если контекста недостаточно или он не подходит, честно скажи об этом.\n"
        "- Если вопрос зависит от предыдущих сообщений и без истории непонятен, попроси уточнить одним вопросом.\n"
        "- Не выдумывай законы, сроки, госпошлины, перечни документов и адреса.\n"
        "- Не упоминай внутренние источники, Qdrant, embeddings или поиск.\n\n"
        f"{ambiguity_note}\n\n"
        "Контекст из базы:\n"
        f"{context_block}\n\n"
        "Формат ответа:\n"
        "1. Сначала короткий человеческий ответ.\n"
        "2. Затем краткое пояснение простыми словами.\n"
        "3. Если точного ответа нет, предложи, что именно стоит уточнить.\n"
    )


def _generate_answer(question: str, context_hits: list[dict[str, Any]]) -> str:
    gigachat, _ = get_neural_models()
    messages = [
        {"role": "system", "content": _build_system_prompt(question=question, context_hits=context_hits)},
        {"role": "user", "content": question},
    ]
    return _normalize_text(gigachat.chat(messages=messages, temperature=0.2))


def _build_top_result(question: str, answer_text: str, context_hits: list[dict[str, Any]]) -> SearchResultItem:
    primary_hit = context_hits[0] if context_hits else None
    answer_name = "Ответ ИИ"
    question_text = question
    if primary_hit:
        if primary_hit["topic"]:
            answer_name = primary_hit["topic"]
        if primary_hit["question"]:
            question_text = primary_hit["question"]

    return SearchResultItem(
        question_id=_stable_id("question", question_text),
        question_text=question_text,
        answer_id=_stable_id("answer", answer_text),
        answer_text=answer_text,
        answer_name=answer_name,
        score=0.99,
    )


def _build_placeholder_results(question: str, top_k: int) -> list[SearchResultItem]:
    placeholders: list[SearchResultItem] = []
    for idx in range(1, top_k):
        placeholder_text = (
            "Чтобы ответить точнее, сформулируйте вопрос подробнее: укажите услугу, документ или жизненную ситуацию."
        )
        placeholders.append(
            SearchResultItem(
                question_id=_stable_id("placeholder-question", f"{question}:{idx}"),
                question_text=question,
                answer_id=_stable_id("placeholder-answer", f"{question}:{idx}"),
                answer_text=placeholder_text,
                answer_name=f"{PLACEHOLDER_TITLE} {idx}",
                score=0.5,
            )
        )
    return placeholders


@app.get("/ping")
def ping() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/search", response_model=list[SearchResultItem])
def search(request: SearchRequest) -> list[SearchResultItem]:
    question = _normalize_text(request.query)
    if not question:
        clarification = "Пожалуйста, уточните вопрос: напишите, какая услуга, документ или ситуация вас интересует."
        return [_build_top_result(question="Уточнение вопроса", answer_text=clarification, context_hits=[])]

    hits = _search_knowledge(question, limit=max(request.top_k, MAX_CONTEXT_HITS))
    context_hits = _select_context_hits(hits)
    answer_text = _generate_answer(question=question, context_hits=context_hits)
    results = [_build_top_result(question=question, answer_text=answer_text, context_hits=context_hits)]

    if RETURN_PLACEHOLDERS and request.top_k > 1:
        results.extend(_build_placeholder_results(question=question, top_k=request.top_k))

    return results


@app.post("/api/update")
def update_question(_: QuestionUpdateRequest):
    return JSONResponse(content={"message": "OK"}, status_code=201)


@app.post("/api/bulk_update")
def bulk_update(_: QuestionBulkUpdateRequest):
    return JSONResponse(content={"message": "OK"}, status_code=201)


@app.post("/api/refresh_faiss")
def refresh_faiss():
    return JSONResponse(content={"message": "OK"}, status_code=201)


@app.post("/api/emb", response_model=EmbeddingResponse)
def get_embedding(request: EmbeddingRequest) -> EmbeddingResponse:
    _, encoder = get_neural_models()
    vector = encoder.get_embedding(_normalize_query(request.query))
    return EmbeddingResponse(embedding=vector, size=len(vector))


@app.get("/api/emb_size")
def get_embedding_size() -> int:
    return VECTOR_SIZE
