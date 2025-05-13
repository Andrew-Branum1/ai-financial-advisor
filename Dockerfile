FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set Python env
ENV DEBIAN_FRONTEND=noninteractive

# Install Python + pip + system deps
RUN apt-get update && apt-get install -y \
    python3 python3-pip git curl unzip build-essential \
    && apt-get clean

# Set working dir
WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Optional: create a placeholder main script
COPY . .

CMD ["bash"]
