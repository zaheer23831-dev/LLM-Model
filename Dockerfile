# CPU-only image for Railway (Railway has no GPUs).
FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.hf_cache \
    PIP_NO_CACHE_DIR=1

# Install the CPU-only build of PyTorch first (much smaller than the CUDA one).
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2.0"

# Install the rest of the dependencies.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the app code (models/ is excluded via .dockerignore and downloaded at runtime).
COPY . .

EXPOSE 8000

# Railway provides $PORT; default to 8000 for local docker runs.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
