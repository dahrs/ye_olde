FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Cloud Run injects $PORT (default 8080) and requires the container to
# listen on it — replaces the fixed 7860 from the original (abandoned, see
# docs §10) Hugging Face Space plan.
ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
