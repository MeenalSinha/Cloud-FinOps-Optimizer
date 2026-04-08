# Root-level Dockerfile — build context is the project root directory.
# Used by automated validators (validate-submission.sh) which run:
#   docker build <repo_dir>
# when a Dockerfile exists at the repo root.
#
# HF Spaces deployment uses server/Dockerfile with the same instructions.

FROM python:3.11-slim

# Create non-root user (uid 1000 — required by Hugging Face Spaces)
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application source — these paths resolve from the project root
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
