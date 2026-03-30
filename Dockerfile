FROM nvidia/cuda:13.0.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev python3-pip \
    git curl build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Build venv at /opt/venv (separate from mounted /workspace)
WORKDIR /opt/build
COPY pyproject.toml uv.lock .python-version ./
COPY constants.py prepare.py train.py ./
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
RUN uv sync --no-dev

# Pre-cache FA3 kernel
RUN uv run python -c "from kernels import get_kernel; k = get_kernel('varunneal/flash-attention-3'); print('FA3:', k)" || echo "FA3 cache done"

# Runtime: /workspace is mounted from host, but uses /opt/venv
WORKDIR /workspace
ENV UV_PROJECT_ENVIRONMENT=/opt/venv

CMD ["uv", "run", "python", "train.py"]
