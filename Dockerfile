FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        libgl1 \
        libglib2.0-0 \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv "${VIRTUAL_ENV}"
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

WORKDIR /app
COPY pyproject.toml README.md ./

RUN python -m pip install --upgrade pip wheel \
    && python -m pip install \
        --index-url https://download.pytorch.org/whl/cu128 \
        torch torchvision \
    && python - <<'PY'
import subprocess
import sys
import tomllib

with open("pyproject.toml", "rb") as file:
    project = tomllib.load(file)["project"]

dependencies = [
    *project["dependencies"],
    *project["optional-dependencies"]["cuda"],
]
dependencies = [
    dependency
    for dependency in dependencies
    if not dependency.startswith(("torch", "torchvision"))
]
subprocess.check_call([sys.executable, "-m", "pip", "install", *dependencies])
PY

COPY src ./src
RUN python -m pip install --no-deps .

EXPOSE 8765

CMD ["uvicorn", "lreader_engine.main:app", "--host", "0.0.0.0", "--port", "8765", "--workers", "1"]
