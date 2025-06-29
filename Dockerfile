# Dockerfile
# Use an official NVIDIA CUDA runtime image as a parent.
# This image contains the necessary CUDA libraries.
FROM nvidia/cuda:12.1.1-base-ubuntu22.04

# Avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.9 and other system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.9 \
    python3.9-dev \
    python3.9-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Make python3.9 the default 'python'
RUN ln -s /usr/bin/python3.9 /usr/bin/python

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Upgrade pip and install requirements.
# By installing torch first with the specific CUDA version, we ensure GPU compatibility.
# The pinned versions in requirements.txt prevent dependency conflicts.
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121 && \
    python -m pip install --no-cache-dir -r requirements.txt

# Download the spaCy model required for preprocessing
RUN python -m spacy download en_core_web_md

# Copy the application source code and assets
COPY ./src ./src
COPY ./static ./static
COPY ./templates ./templates

# Expose port 8000
EXPOSE 8000

# Define the default command to run the web application
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]