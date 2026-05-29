FROM python:3.11-slim

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python deps (cached layer) ────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── App code ──────────────────────────────────────────────────────────────────
COPY . .

# ── Create runtime dirs ───────────────────────────────────────────────────────
RUN mkdir -p data models vectorstore

# ── Streamlit config ──────────────────────────────────────────────────────────
RUN mkdir -p .streamlit && cat > .streamlit/config.toml <<EOF
[server]
port = 8501
address = "0.0.0.0"
headless = true

[theme]
base = "dark"
primaryColor = "#0EA5E9"
backgroundColor = "#0F172A"
secondaryBackgroundColor = "#1E293B"
textColor = "#E2E8F0"
EOF

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
