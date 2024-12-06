FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/processed/predictions

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
