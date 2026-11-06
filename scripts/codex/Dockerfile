FROM node:24-slim
RUN apt-get update && apt-get install -y --no-install-recommends ripgrep ca-certificates python3 && rm -rf /var/lib/apt/lists/*
ARG CODEX_VERSION=0.160.0
RUN npm install -g @openai/codex@${CODEX_VERSION}
WORKDIR /docs
