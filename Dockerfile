FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
COPY ./app/app.py .
COPY gigachat-ca.pem .
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY app/ /app/
COPY certs/ /app/certs/
COPY data/ /app/data/

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]