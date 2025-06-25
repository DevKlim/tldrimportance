# Use an official NVIDIA CUDA runtime image as a parent.
# This image contains the necessary CUDA libraries.
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python, pip, and other common tools
RUN apt-get update && \
    apt-get install -y python3.9 python3-pip python3.9-venv && \
    rm -rf /var/lib/apt/lists/*

# Make python3.9 the default 'python'
RUN ln -s /usr/bin/python3.9 /usr/bin/python

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the specific GPU version of PyTorch as requested.
# This is done before other requirements to ensure compatibility.
RUN python -m pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install the rest of the packages from requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy the application source code, static files, and templates
COPY ./src ./src
COPY ./static ./static
COPY ./templates ./templates

# Make port 8000 available
EXPOSE 8000

# Define the default command to run the application
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]