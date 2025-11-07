FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir httpx pytest
ENV PYTHONUNBUFFERED=1
CMD ["python","-m","services.features.function_ssrf"]
