# Container image for the FastAPI serving layer (course week 4: Docker).
# Build:  docker build -t predmaint-api .
# Run:    docker run -p 8000:8000 -e API_KEY=change-me \
#               -v "$PWD/mlruns:/app/mlruns" -v "$PWD/mlflow.db:/app/mlflow.db" \
#               predmaint-api
FROM python:3.11-slim

WORKDIR /app

# Install only what the serving layer needs (keeps the image small).
COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi==0.128.8 uvicorn==0.39.0 \
    mlflow==2.17.2 scikit-learn==1.5.2 pandas==2.2.3 numpy==1.26.4 \
    python-dotenv==1.0.1

COPY src/ ./src/
COPY reports/registered_model.json ./reports/registered_model.json

ENV PYTHONPATH=/app/src
EXPOSE 8000

CMD ["uvicorn", "api_service:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "src"]
