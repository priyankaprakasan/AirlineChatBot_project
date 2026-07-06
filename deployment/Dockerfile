# Use a slim Python base image
FROM python:3.10-slim

# Install system dependencies required for llama-cpp-python and pypdf
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir keeps the image size smaller
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY app.py .

# Hugging Face Spaces requires the app to listen on port 7860
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]
