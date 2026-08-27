FROM python:3.11-slim AS native-builder

RUN apt-get update && apt-get install -y --no-install-recommends curl build-essential \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal \
    && pip install --no-cache-dir 'maturin>=1.7,<2' \
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/root/.cargo/bin:${PATH}"
COPY native/solana_fastpath /build/solana_fastpath
RUN maturin build --release --manifest-path /build/solana_fastpath/Cargo.toml --out /wheels

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --from=native-builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels

COPY src/ ./src/
COPY config/ ./config/
COPY tests/ ./tests/

RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin app \
    && mkdir -p /app/data/launch_episodes /app/logs /app/models \
    && chown -R app:app /app

ENV PYTHONPATH=/app

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

EXPOSE 8080

USER app

CMD ["python", "-m", "src.main", "--dry-run"]
