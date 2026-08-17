# syntax=docker/dockerfile:1

# hatch-vcs derives the version from git metadata, so the build stage needs
# both git itself and the .git directory that .dockerignore deliberately keeps.
FROM python:3.12-slim AS build
RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN pip install --no-cache-dir build \
 && python -m build --wheel --outdir /dist


FROM python:3.12-slim

# onnxruntime, which FastEmbed loads for the embedding model, links libgomp.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    HOME=/home/smrti \
    SMRTI_DB=/data/memory.db \
    FASTEMBED_CACHE_PATH=/opt/smrti/models

RUN useradd --system --create-home --home-dir /home/smrti --shell /usr/sbin/nologin smrti \
 && mkdir -p /data /opt/smrti/models \
 && chown smrti:smrti /data /opt/smrti/models

COPY --from=build /dist /tmp/dist
RUN WHEEL="$(ls /tmp/dist/*.whl)" \
 && pip install --no-cache-dir "${WHEEL}[openai]" \
 && rm -rf /tmp/dist

USER smrti

# Bake the ~120MB embedding model into the image: without it the first recall
# stalls on a HuggingFace download, and an offline host never recalls at all.
# Runs as the runtime user so the files land owned by it — a later chown -R
# would copy every one of them into a second layer. gliner2 is deliberately
# left out: it pulls torch, and extraction falls back to LLM-only without it.
RUN python -c "from smrti.core.embed import get_embedding_provider; get_embedding_provider().embed('warmup')"

VOLUME ["/data"]
EXPOSE 8420

# Tracks the port in the default CMD; override both together to serve elsewhere.
# A bare TCP connect, because /status answers 401 once SMRTI_API_KEY is set.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import socket; socket.create_connection(('127.0.0.1', 8420), 3).close()"

ENTRYPOINT ["smrti"]
CMD ["serve", "rest", "--host", "0.0.0.0", "--port", "8420"]
