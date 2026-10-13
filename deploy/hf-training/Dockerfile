FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    git gcc g++ libgomp1 ca-certificates wget \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 1000 user
USER user
ENV PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /home/user/app
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
COPY --chown=user:user start.py check_environment.py gpu_smoke_test.py ./
EXPOSE 7860
CMD ["python", "start.py"]
