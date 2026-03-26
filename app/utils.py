import os
import logging
from openai import OpenAI
from openai import (
    APIError,
    AuthenticationError,
    PermissionDeniedError,
)

from sentence_transformers import SentenceTransformer




logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("gigachat")



class GigaChatClient:
    def __init__(self):
        logger.info("Инициализация GigaChat клиента")

        api_key = os.getenv("API_KEY")
        if not api_key:
            logger.critical("API_KEY не задан в окружении")
            raise RuntimeError("API_KEY не задан в окружении")

        self.base_url = os.getenv(
            "GIGACHAT_BASE_URL",
            "https://foundation-models.api.cloud.ru/v1"
        )

        self.model = os.getenv(
            "GIGACHAT_MODEL",
            "ai-sage/GigaChat3-10B-A1.8B"
        )

        logger.info(f"Base URL: {self.base_url}")
        logger.info(f"Model: {self.model}")

        self.client = OpenAI(
            api_key=api_key,
            base_url=self.base_url
        )

        logger.info("GigaChat клиент успешно создан")

    def healthcheck(self) -> bool:
        """
        Проверка доступности сервиса и авторизации
        """
        logger.info("Запуск healthcheck GigaChat")

        try:
            self.chat(
                messages=[{"role": "user", "content": "ping"}],
                temperature=0.0
            )
            logger.info("GigaChat healthcheck: OK")
            return True

        except Exception:
            logger.error("GigaChat healthcheck: FAILED")
            return False

    def chat(self, messages, temperature: float = 0.1) -> str:
        logger.info("Отправка запроса в GigaChat")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=2500,
                top_p=0.95,
                presence_penalty=0
            )

            logger.info("Запрос к GigaChat выполнен успешно (200 OK)")
            return response.choices[0].message.content

        except AuthenticationError as e:
            logger.error("Ошибка авторизации (401 Unauthorized)")
            logger.error(str(e))
            raise

        except PermissionDeniedError as e:
            logger.error("Доступ запрещён (403 Forbidden)")
            logger.error(str(e))
            raise

        except APIError as e:
            status = getattr(e, "status_code", "unknown")
            logger.error(f"Ошибка API GigaChat (HTTP {status})")
            logger.error(str(e))
            raise

        except Exception as e:
            logger.exception("Неизвестная ошибка при обращении к GigaChat")
            raise


class LocalEncoder:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            logger.info("Создание LocalEncoder (Singleton)")
            cls._instance = super().__new__(cls)

            logger.info("Загрузка модели векторизации MiniLM")
            cls._model = SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
            logger.info("Модель MiniLM успешно загружена")

        return cls._instance

    def get_embedding(self, text: str) -> list:
        logger.debug("Вычисление embedding для текста")
        return self._model.encode(text).tolist()
