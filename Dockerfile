FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ /app/
COPY certs/ /app/certs/
COPY data/ /app/data/

EXPOSE 8501

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8501"]
