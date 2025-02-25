# Use Ubuntu 22.04 with CUDA support
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Set environment variables
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

# Install required packages
RUN apt update && apt install -y \
    wget \
    python3-dev \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir -p /root/.conda \
    && sh Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

# Create a Conda environment and install dependencies
RUN conda create -y -n ml python=3.10 && \
    echo "source activate ml" > ~/.bashrc

# Copy project files
COPY . /src/
WORKDIR /src/

# Install dependencies
RUN /bin/bash -c "source activate ml && pip install -r requirements.txt"

# Define entrypoint
CMD ["python", "app.py"]
