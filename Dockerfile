FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.8.2

# Install Poetry
RUN pip install "poetry==$POETRY_VERSION"

# Set up app directory
WORKDIR /app

# Copy files and install dependencies
COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-interaction --no-ansi

# Copy the rest of the project
COPY . .

# Use all CPU cores by default
ENV NUM_WORKERS=20

# Set default command
ENTRYPOINT ["poetry", "run", "python", "scripts/launch_jobs.py"]