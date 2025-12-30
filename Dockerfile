FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (tesseract optional; remove if you don't need OCR)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# Cloud Run sets $PORT. Streamlit must listen on it.
CMD ["bash", "-lc", "streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT --server.headless=true"]

