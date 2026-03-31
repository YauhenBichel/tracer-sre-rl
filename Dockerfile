FROM python:3.11-slim

WORKDIR /project

COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY app/ app/
COPY scenarios/ scenarios/
COPY config/ config/
COPY data/ data/
COPY tests/ tests/

CMD ["python", "-m", "app.main"]
