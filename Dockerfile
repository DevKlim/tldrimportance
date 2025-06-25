# Dockerfile
# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir ensures the image is smaller
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source code, static files, and templates
COPY ./src ./src
COPY ./static ./static
COPY ./templates ./templates

# CRITICAL: Copy the pre-trained model into the container.
# This assumes you have run `python src/train.py` locally first,
# which creates the ./results directory.
COPY ./results ./results

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define the command to run the application
# This tells uvicorn to run the 'app' instance from the 'src.app' module.
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]