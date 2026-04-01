FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir openai-whisper watchdog

WORKDIR /app

COPY transcribe_watch.py /app/transcribe_watch.py

CMD ["python", "/app/transcribe_watch.py"]
