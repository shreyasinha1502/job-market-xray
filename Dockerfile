# Render web service: Gradio dashboard + int8 ONNX seniority classifier (CPU, fits 512 MB).
# Data comes from the committed data/processed (refreshed daily by GitHub Actions); model weights
# are downloaded once at startup from MODEL_URL (a GitHub release asset), never baked into git.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    PYTHONPATH=/app/src \
    XRAY_MODEL_DIR=/app/models

WORKDIR /app
COPY requirements-app.txt .
RUN pip install -r requirements-app.txt

COPY config ./config
COPY src ./src
COPY data/processed ./data/processed

RUN useradd --create-home app && mkdir -p /app/models && chown -R app /app
USER app

EXPOSE 10000
CMD ["python", "-m", "xray.app"]
