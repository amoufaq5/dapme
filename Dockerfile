# Dockerfile

FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a working directory
WORKDIR /app

# Copy the requirements
COPY api/requirements.txt /app/api_requirements.txt
RUN pip install --no-cache-dir -r /app/api_requirements.txt

# Copy all source code
COPY api /app/api
COPY model_training /app/model_training

# Copy any model weights (assuming they're in model_training or somewhere)
# If you store them separately, you'd fetch from GCS or similar in production
# e.g. COPY model_weights /app/model_weights

# Expose port
EXPOSE 8080

# By default, run the API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
