FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY scenarios/ scenarios/
COPY config/ config/
COPY demo.py .
COPY run_crawler.py .
COPY run_training.py .
COPY tests/ tests/

# Default: run the demo
CMD ["python", "demo.py"]
