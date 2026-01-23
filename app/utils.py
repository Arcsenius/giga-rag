import os
from openai import OpenAI


class GigaChatClient:
    def __init__(self):
        api_key = os.getenv("API_KEY")
        if not api_key:
            raise RuntimeError("API_KEY не задан в окружении")

        base_url = os.getenv(
            "GIGACHAT_BASE_URL",
            "https://foundation-models.api.cloud.ru/v1"
        )

        self.model = os.getenv(
            "GIGACHAT_MODEL",
            "ai-sage/GigaChat3-10B-A1.8B"
        )

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def chat(self, messages, temperature=0.1):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=2500,
                top_p=0.95,
                presence_penalty=0
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"Ошибка LLM API: {e}"


from sentence_transformers import SentenceTransformer

class LocalEncoder:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LocalEncoder, cls).__new__(cls)
            print("Загрузка модели векторизации (MiniLM)...")
            cls._model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            print("Модель готова.")
        return cls._instance

    def get_embedding(self, text: str) -> list:
        return self._model.encode(text).tolist()
