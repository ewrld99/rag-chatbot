# Use official Python lightweight image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies (required for some Python packages like psycopg2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire backend source code
# Note: app/, scripts/, alembic/, and alembic.ini will be copied.
# .env, uploads/, and frontend are excluded via .dockerignore
COPY . .

# Create the uploads directory inside the container (this will be mapped to a volume)
RUN mkdir -p uploads

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the application using Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
