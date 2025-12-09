# Use an official Python runtime as the base image
FROM python:3.9-slim

# Expose the port Streamlit runs on
EXPOSE 8501

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install system dependencies required for some Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Get Updates
RUN apt-get update

# Install Python dependencies
RUN pip install -r requirements.txt

# Install Requirements
RUN pip3 install -r requirements.txt
COPY . .

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Command to run the Streamlit app
ENTRYPOINT ["streamlit", "run"] 
CMD ["inference.py"]
