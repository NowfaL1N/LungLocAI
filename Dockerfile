# LungLocAI web app — runs on Hugging Face Spaces (Docker SDK), Render, Railway, or any Docker host.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=user . .
# bundled sample scans for the "try a sample" buttons
RUN python tools/make_synthetic_ct.py --out data/samples --n 1 --seed 101 \
 && python tools/make_synthetic_ct.py --out data/samples --n 1 --full-body --seed 202 \
 && mkdir -p data/jobs && chown -R user:user data

USER user
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860} --proxy-headers --forwarded-allow-ips='*'"]
