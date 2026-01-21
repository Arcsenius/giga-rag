import os
import requests
import time
from uuid import uuid4
from pathlib import Path
from sentence_transformers import SentenceTransformer

CA_CERT_PATH = Path("/app/certs/gigachat-ca.pem")
GIGACHAT_AUTH = os.getenv("GIGACHAT_AUTH") 

OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
CHAT_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
MODEL_NAME = "GigaChat-2-Max"

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

class GigaChatClient:
    def __init__(self):
        self.access_token = None
        self.token_expires_at = 0

    def _refresh_token(self):
        rq_uid = str(uuid4())
        headers = {
            "Authorization": f"Basic {GIGACHAT_AUTH}",
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": rq_uid
        }
        try:
            resp = requests.post(OAUTH_URL, headers=headers, data="scope=GIGACHAT_API_B2B", verify=str(CA_CERT_PATH), timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                self.access_token = data["access_token"]
                self.token_expires_at = (data["expires_at"] / 1000) - 60
                return True
            print(f"Auth Error: {resp.text}")
            return False
        except Exception as e:
            print(f"Auth Exception: {e}")
            return False

    def get_token(self):
        if not self.access_token or time.time() > self.token_expires_at:
            if not self._refresh_token():
                raise Exception("Не удалось получить токен GigaChat")
        return self.access_token

    def chat(self, messages, temperature=0.1):
        try:
            token = self.get_token()
            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
            payload = {
                "model": MODEL_NAME,
                "messages": messages,
                "temperature": temperature
            }
            resp = requests.post(CHAT_URL, headers=headers, json=payload, verify=str(CA_CERT_PATH), timeout=30)
            
            if resp.status_code == 401:
                self._refresh_token()
                token = self.get_token()
                headers["Authorization"] = f"Bearer {token}"
                resp = requests.post(CHAT_URL, headers=headers, json=payload, verify=str(CA_CERT_PATH), timeout=30)

            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            return f"Ошибка API: {resp.status_code} {resp.text}"
        except Exception as e:
            return f"Ошибка запроса: {e}"