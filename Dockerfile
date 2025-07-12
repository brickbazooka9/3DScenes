FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Install Python and pip
RUN apt-get update && \
    apt-get install -y python3 python3-pip git && \
    apt-get clean

# Upgrade pip
RUN pip3 install --upgrade pip

# Copy your requirements file into the image
COPY requirements.txt .

# Install required Python packages
RUN pip3 install -r requirements.txt

# Set the working directory
WORKDIR /workspace
