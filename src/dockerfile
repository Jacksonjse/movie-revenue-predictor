# Dockerfile (place at repo root)
FROM python:3.10-slim

WORKDIR /app

# System deps (if needed)
RUN apt-get update && apt-get install -y build-essential gcc && rm -rf /var/lib/apt/lists/*

# copy code
COPY src/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

COPY src /app/src
WORKDIR /app/src

# expose port
ENV PORT 8000
EXPOSE 8000

# Ensure the model exists — your pipeline.joblib should be created by running train_model.py locally
# The container expects src/model/pipeline.joblib to exist (you could also train on build, but training in CI is heavy)
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8000", "app:app", "--timeout", "120"]
