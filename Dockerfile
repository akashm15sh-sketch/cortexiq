# CortexIQ Neural Signal Platform — Production Docker Image
FROM python:3.12-slim

# System dependencies for MNE, matplotlib, and scientific computing
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ gfortran \
    libhdf5-dev \
    libfreetype6-dev \
    libpng-dev \
    libjpeg-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create required directories
RUN mkdir -p data/uploads results

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg
ENV MNE_LOGGING_LEVEL=WARNING

# Expose port (Render sets $PORT automatically)
EXPOSE 7860

# Start command — Render injects $PORT
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
