# # Start from an official Python base image
# # Using a 'slim' variant keeps the final image size smaller.
# FROM python:3.11-slim

# # Set the working directory inside the container
# WORKDIR /app

# # Copy only the requirements file first and install dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy the rest of your application's code and data into the container
# COPY . .

# # Expose the port the app runs on
# # This tells Docker that the container listens on port 80.
# EXPOSE 80


# # The host must be 0.0.0.0 to be accessible from outside the container.
# CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "80"]
# Use multi-stage build
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.11-slim

# Create non-root user
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Copy only necessary files
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=80

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:80/health || exit 1

# Run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "80"]