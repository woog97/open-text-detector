FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY vendor ./vendor
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && pip install --no-cache-dir uv && uv sync --frozen --no-dev \
    && apt-get purge -y --auto-remove build-essential && rm -rf /var/lib/apt/lists/*
COPY open_text_detector_backend ./open_text_detector_backend
COPY migrations ./migrations
RUN useradd --uid 10001 --create-home workbench && mkdir /data && chown workbench:workbench /data
USER workbench
ENV OTD_DATA_DIR=/data
EXPOSE 8000
CMD [".venv/bin/python", "-m", "open_text_detector_backend", "serve", "--host", "0.0.0.0"]
