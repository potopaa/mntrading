FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_MAX_THREADS=8 \
    TZ=Europe/Kyiv

WORKDIR /app

# sys deps (lightgbm runtime, git, TLS)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    ca-certificates \
    git \
 && rm -rf /var/lib/apt/lists/*

# deps layer
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# project
COPY . /app

# run command set in docker-compose (container stays alive)
CMD ["bash", "-lc", "tail -f /dev/null"]
