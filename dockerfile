# Dockerfile (place at repo root)
FROM python:3.10-slim

WORKDIR /app

# System deps (if needed)
RUN apt-get update && apt-get install -y build-essential gcc && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY src/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

# Copy source code and model
COPY src /app/src
COPY model /app/model

# Set working directory
WORKDIR /app/src

# Expose port
ENV PORT 8000
EXPOSE 8000

# Start the app
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8000", "app:app", "--timeout", "120"]
