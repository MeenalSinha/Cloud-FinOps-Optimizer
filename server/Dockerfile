# server/Dockerfile — used by HF Spaces deployment.
# IMPORTANT: must be built from the PROJECT ROOT directory, not from server/.
#   Correct:   docker build -f server/Dockerfile .
#   Incorrect: docker build server/   (build context wrong — COPY will fail)
#
# The root Dockerfile (../Dockerfile) is identical and is found automatically
# by tools that look for a Dockerfile at the project root.

FROM python:3.11-slim

# Create non-root user (uid 1000 — required by Hugging Face Spaces)
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application source files from project root
COPY models.py .
COPY client.py .
COPY inference.py .
COPY openenv.yaml .
COPY pyproject.toml .
COPY validate.py .
COPY README.md .
COPY server/ ./server/

# Ensure server/ is a proper Python package
RUN touch server/__init__.py

# Hugging Face Spaces uses port 7860
ENV PORT=7860
ENV HOST=0.0.0.0
ENV WORKERS=1

USER appuser

EXPOSE 7860

CMD ["sh", "-c", "uvicorn server.app:app --host $HOST --port $PORT --workers $WORKERS"]
