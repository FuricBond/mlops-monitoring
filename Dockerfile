# Use an official Python 3.11 slim base image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set /app as the working directory inside the container
WORKDIR /app

# Copy dependency specifications first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies without pip caching to minimize image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy FastAPI application script
COPY app.py .

# Copy model artifacts and training metadata
COPY model/ ./model/

# Expose container port 8000
EXPOSE 8000

# Start FastAPI application using Uvicorn on 0.0.0.0:8000 with a single worker and no reload
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
