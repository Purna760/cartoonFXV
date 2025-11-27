FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY render_requirements.txt .
RUN pip install --no-cache-dir -r render_requirements.txt

COPY . .

RUN mkdir -p static/uploads static/processed

EXPOSE 10000

CMD ["gunicorn", "--config", "gunicorn_render.conf.py", "main:app"]
