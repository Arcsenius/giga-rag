import streamlit as st
import os
from qdrant_client import QdrantClient
from utils import GigaChatClient, LocalEncoder

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "knowledge_base"

st.set_page_config(page_title="Корпоративный Бот", page_icon="🤖")
st.title("🤖 База знаний 2025")

@st.cache_resource
def get_neural_models():
    gc = GigaChatClient()
    enc = LocalEncoder()
    return gc, enc

def get_db_client():
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

try:
    gigachat, encoder = get_neural_models()
    qdrant = get_db_client()
except Exception as e:
    st.error(f"Ошибка инициализации: {e}")
    st.stop()

def contextualize_query(user_input: str, history: list) -> str:
    """
    Превращает 'А сколько она стоит?' в 'Сколько стоит карта болельщика?'
    используя историю диалога.
    """
    if not history:
        return user_input

    short_history = history[-3:] 
    
    prompt = (
        "Ты — технический модуль переформулирования поисковых запросов. "
        "Твоя задача: переписать последний вопрос пользователя так, чтобы он стал полным и понятным без контекста беседы. "
        "Замени местоимения ('она', 'это', 'там') на конкретные термины из истории. "
        "НЕ ОТВЕЧАЙ на вопрос. ПРОСТО ВЕРНИ ПЕРЕФОРМУЛИРОВАННЫЙ ВОПРОС."
        "\n\nИстория диалога:"
    )
    
    for msg in short_history:
        role = "Пользователь" if msg["role"] == "user" else "Ассистент"
        prompt += f"\n{role}: {msg['content']}"
    
    prompt += f"\nПользователь (последний вопрос): {user_input}"
    prompt += "\n\nПереформулированный поисковый запрос:"

    try:
        messages = [{"role": "user", "content": prompt}]
        rewritten_query = gigachat.chat(messages, temperature=0.1)
        
        return rewritten_query.strip().replace('"', '').replace("'", "")
    except:
        return user_input # 

def get_context_from_db(query: str, limit=3):
    try:
        normalized_query = query.strip().lower()
        query_vector = encoder.get_embedding(normalized_query)
        
        SCORE_THRESHOLD = 0.36 

        groups = qdrant.query_points_groups(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            group_by="answer_hash",
            limit=limit,
            with_payload=True,
            group_size=1
        ).groups

        context_parts = []
        
        with st.expander("🕵️ DEBUG: Поиск в базе"):
            if not groups: st.write("База не вернула результатов.")
            for g in groups:
                hit = g.hits[0]
                is_good = hit.score >= SCORE_THRESHOLD
                icon = "✅" if is_good else "❌"
                st.write(f"Score: **{hit.score:.4f}** {icon}")
                st.caption(hit.payload.get('topic', 'Без темы'))

        valid_hits_count = 0
        for g in groups:
            best_hit = g.hits[0]
            if best_hit.score < SCORE_THRESHOLD:
                continue
            
            p = best_hit.payload
            valid_hits_count += 1
            context_parts.append(
                f"=== Блок информации (Тема: {p.get('topic', 'Общее')}) ===\n"
                f"Ответ: {p.get('answer', '')}"
            )
        
        if valid_hits_count == 0:
            return ""
            
        return "\n\n".join(context_parts)

    except Exception as e:
        st.error(f"Ошибка поиска: {e}")
        return ""

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Задайте вопрос..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Думаю..."):
            
            
            search_query = user_input
            if len(st.session_state.messages) > 1:
                with st.status("Уточняю контекст вопроса...", expanded=False) as status:
                    search_query = contextualize_query(user_input, st.session_state.messages[:-1])
                    status.write(f"Оригинал: {user_input}")
                    status.write(f"Поиск по: **{search_query}**")
                    status.update(label="Контекст уточнен", state="complete")
            
            context = get_context_from_db(search_query)
            
            if context:
                context_instruction = (
                    "У тебя есть официальная информация из базы знаний:\n"
                    f"{context}\n"
                    "Используй её как основную и достоверную."
                )
            else:
                context_instruction = (
                    "В базе знаний нет точной информации по этому вопросу.\n"
                    "Объясни это пользователю вежливо и человечески, "
                    "предложи возможные общие варианты и порекомендуй, "
                    "куда можно обратиться за уточнением."
                )


            system_prompt = (
    "Ты — дружелюбный, понятный и компетентный консультант центра «Мои Документы» (МФЦ). "
    "Ты объясняешь так, как если бы разговаривал с обычным человеком, "
    "который не знает юридических и бюрократических терминов.\n\n"

    "ОБЯЗАТЕЛЬНАЯ СТРУКТУРА ОТВЕТА:\n"
    "1. Короткий человеческий ответ (1–2 предложения, без канцелярита)\n"
    "2. Подробное объяснение простыми словами\n"
    "3. Что нужно сделать человеку (пошагово, если применимо)\n"
    "4. Важные нюансы / ограничения (если есть)\n\n"

    "СТРОГИЕ ПРАВИЛА:\n"
    "- НЕ сокращай ответ\n"
    "- НЕ используй сухой бюрократический стиль\n"
    "- Пиши развернуто, но понятно\n"
    "- Если используешь термины — объясняй их\n"
    "- Если информации нет в контексте — честно скажи об этом\n\n"

    "РАБОТА С КОНТЕКСТОМ:\n"
    "• Если КОНТЕКСТ есть — опирайся ТОЛЬКО на него\n"
    "• Если вопрос общий — сначала дай определение, затем объяснение\n"
    "• Если вопрос конкретный — отвечай строго по фактам из КОНТЕКСТА\n\n"

    f"{context_instruction}"
)

            
            
            messages_payload = (
            [{"role": "system", "content": system_prompt}] +
            [{"role": "assistant", "content": "Я консультант МФЦ и объясняю простым языком."}] +
            st.session_state.messages
            )

            
            response = gigachat.chat(
            messages_payload,
            temperature=0.6
            )


            st.markdown(response)
            
            with st.expander("🔍 Техническая информация"):
                st.write(f"**Поисковый запрос:** {search_query}")
                st.text(context if context else "Нет релевантного контекста")

    st.session_state.messages.append({"role": "assistant", "content": response})