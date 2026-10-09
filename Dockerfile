# What changed in the room? — GPU image.
#
#   docker build -t changedet .
#   docker run --gpus all -u "$(id -u):$(id -g)" -e HOME=/tmp --env-file .env \
#     -v "$PWD/checkpoints:/app/checkpoints" -v "$PWD/runs:/app/runs" -v /path/to/recordings:/data \
#     changedet all --a /data/vid1 --b /data/vid2 --run demo
#
# Model checkpoints are downloaded on first use into the mounted checkpoints/ folder.
# --env-file .env passes ANTHROPIC_API_KEY for the optional C7 visual check (omit to skip it).
# PyTorch wheels bundle their own CUDA libraries; the host needs an NVIDIA driver >= 580.
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgl1 libegl1 libglib2.0-0 libgomp1 libusb-1.0-0 libtbb12 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.12.24 /uv /uvx /bin/

ENV UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_COMPILE_BYTECODE=1 \
    MPLBACKEND=Agg
WORKDIR /app

# Dependencies first (cached layer), then the project itself.
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY changedet ./changedet
COPY configs ./configs
COPY scripts ./scripts
RUN uv sync --frozen --no-dev

ENTRYPOINT ["uv", "run", "--no-sync", "python", "-m", "changedet.cli"]
CMD ["--help"]
