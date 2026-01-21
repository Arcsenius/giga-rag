import os
import hashlib
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from utils import LocalEncoder

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "knowledge_base"
VECTOR_SIZE = 384

DATA_PATH = "/app/data/ВОП2025.xlsx"

def ingest():
    print(f"Начинаем чтение {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        print("Файл Excel не найден.")
        return

    df = pd.read_excel(DATA_PATH).fillna("")
    print(f"Найдено {len(df)} строк.")

    encoder = LocalEncoder()
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="answer_hash",
        field_schema="keyword"
    )
    print("Коллекция создана + Индекс по answer_hash.")

    points = []
    for idx, row in df.iterrows():
        topic = str(row.get("Название вопроса", "")).strip()
        question = str(row.get("Текст вопроса", "")).strip()
        answer = str(row.get("Текст ответа", "")).strip()
        
        text_to_vectorize = f"{topic} {question}".strip().lower()
        vector = encoder.get_embedding(text_to_vectorize)
        
        
        answer_normalized = " ".join(answer.split()) 
        answer_hash = hashlib.md5(answer_normalized.encode('utf-8')).hexdigest()

        payload = {
            "topic": topic,
            "question": question, 
            "answer": answer,
            "answer_hash": answer_hash 
        }
        
        points.append(PointStruct(id=idx, vector=vector, payload=payload))

        if idx % 100 == 0:
            print(f"🔹 {idx}...")

    if points:
        client.upload_points(collection_name=COLLECTION_NAME, points=points)
        print(f"Загружено {len(points)} векторов.")

if __name__ == "__main__":
    ingest()