# Start from an official Python base image
# Using a 'slim' variant keeps the final image size smaller.
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy only the requirements file first and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code and data into the container
COPY . .

# Expose the port the app runs on
# This tells Docker that the container listens on port 80.
EXPOSE 80


# The host must be 0.0.0.0 to be accessible from outside the container.
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "80"]
