# Base image
FROM python:3.11.1-slim

# Set working directory
WORKDIR /app

# Install Java (needed for H2O) and clean up cache
RUN apt-get update && apt-get install -y --no-install-recommends default-jre \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install core packages
RUN python -m pip install --upgrade pip setuptools wheel

# Install H2O (stable version)
RUN pip install https://h2o-release.s3.amazonaws.com/h2o/rel-3.46.0/7/Python/h2o-3.46.0.7-py2.py3-none-any.whl

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip cache purge

# Copy the rest of your app
COPY . .

# Expose ports for:
# - H2O Flow (web GUI): 54321
# - Streamlit UI: 8501
EXPOSE 54321
EXPOSE 8501

# Start Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

