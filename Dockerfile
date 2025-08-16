FROM python:3.12-slim AS base

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

# системные зависимости для numpy/pandas/pyarrow/tqdm/lightgbm и пр.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        curl ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/*

# отдельный слой для зависимостей
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
 && pip install -r requirements.txt

# копируем весь проект
COPY . /app

# по умолчанию контейнер ничего не запускает — команду задаём в docker-compose
# EXPOSE делаем в compose
