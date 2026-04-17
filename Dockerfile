FROM python:3.12-slim

WORKDIR /app

# Build tools needed by some Python C extensions (spaCy, Pillow)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first — Docker layer cache optimisation
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install the reskillio package in editable mode
COPY pyproject.toml .
COPY reskillio/ reskillio/
COPY config/ config/
RUN pip install --no-cache-dir -e .

ENV PYTHONPATH=/app

# KFP components run as the default user — no CMD needed
