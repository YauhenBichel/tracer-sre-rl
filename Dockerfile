FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY scenarios/ scenarios/
COPY config/ config/
COPY data/ data/
COPY demo.py .
COPY run_crawler.py .
COPY run_training.py .
COPY Makefile .
COPY tests/ tests/

CMD ["python", "demo.py", "--quiet"]
