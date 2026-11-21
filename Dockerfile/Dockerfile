FROM ghcr.io/astral-sh/uv:0.13.7 AS uv

# Also usable to generate uv.lock before building the final image.
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04 AS uv-base
COPY --from=uv /uv /uvx /usr/local/bin/
ENV DEBIAN_FRONTEND=noninteractive \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    PATH=/opt/venv/bin:${PATH} \
    PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates python3 python3-venv python3-dev \
        build-essential git openssh-client libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace

FROM uv-base AS final
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked && uv pip check --python /opt/venv/bin/python \
    && useradd --create-home --shell /bin/bash --no-log-init mlcycle \
    && mkdir -p /data \
    && chown -R mlcycle:mlcycle /workspace /data /opt/venv
COPY --chown=mlcycle:mlcycle . .
USER mlcycle
ENV HOME=/home/mlcycle
CMD ["uv", "run", "pytest"]
