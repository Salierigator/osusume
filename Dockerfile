# Backend image
FROM python:3.11-slim

# LightGBM runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1
WORKDIR $HOME/app

# Dependencies
COPY --chown=user requirements-deploy.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 && \
    pip install --no-cache-dir -r requirements-deploy.txt && \
    pip install --no-cache-dir huggingface_hub==0.36.0

# Model bundle from Hugging Face
ARG ARTIFACTS_REPO=Salierigator/anime-recommender-artifacts
ENV ARTIFACTS_REPO=${ARTIFACTS_REPO}
RUN if [ -n "$ARTIFACTS_REPO" ]; then \
      python -c "import os; from huggingface_hub import snapshot_download; \
snapshot_download(repo_id=os.environ['ARTIFACTS_REPO'], local_dir='.', \
allow_patterns=['artifacts/*', 'data/cleaned/*'])" && rm -rf .cache && \
      mkdir -p export/twotower && mv artifacts/* data/cleaned/* export/twotower/ && \
      rmdir artifacts data/cleaned data ; \
    else echo '>> skip the bundle download: expect a bind mount of export/twotower/ at run time' ; fi

# Code
COPY --chown=user . .

# Server
ENV MOCK_MODE=0
WORKDIR $HOME/app/service/backend
EXPOSE 7860
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}
