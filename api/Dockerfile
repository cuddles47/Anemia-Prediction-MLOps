FROM python:3.10-slim

WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/ /app/api/

# Set environment variables
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

# Set the API port
EXPOSE 5001

# Start the application
ENTRYPOINT ["python", "-m", "api.serve_model", "--host", "0.0.0.0", "--port", "5001"]