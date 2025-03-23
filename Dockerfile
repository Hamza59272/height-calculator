# Use Python 3.10.12 as the base image for building dependencies
FROM python:3.10.12 AS builder

# Set environment variables for Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set the working directory inside the container
WORKDIR /app

# Create a virtual environment
RUN python -m venv /app/.venv

# Copy only requirements to leverage Docker cache
COPY requirements.txt ./

# Install dependencies inside the virtual environment
RUN /app/.venv/bin/pip install --no-cache-dir -r requirements.txt

# Use a lightweight Python image for the final stage
FROM python:3.10.12-slim

# Set working directory
WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy the rest of the application code
COPY . .

# Expose the FastAPI port (default is 8000)
EXPOSE 8000

# Set the command to run FastAPI with Uvicorn
CMD ["/app/.venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
