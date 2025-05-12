FROM python:3.10-slim

# Установка зависимостей (git может понадобиться Detoxify)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY toxicity_service.py ./

EXPOSE 5001

CMD ["python", "toxicity_service.py"]
