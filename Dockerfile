FROM python:3.11-slim

WORKDIR /app

# System deps for pygame (SDL)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libSDL2-2.0-0 libSDL2-image-2.0-0 libSDL2-mixer-2.0-0 libSDL2-ttf-2.0-0 \
    libfreetype6 libjpeg62-turbo \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "exploration.train_all"]

