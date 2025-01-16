# Use Ubuntu 20.04 as the base image
FROM ubuntu:20.04

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    python3 \
    python3-pip \
    cmake \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/miniconda \
    && rm /tmp/miniconda.sh \
    && /opt/miniconda/bin/conda init

# Update PATH to include Miniconda
ENV PATH="/opt/miniconda/bin:$PATH"

# Set the working directory
WORKDIR /workspace

# Clone the ExecuTorch repository and switch to the release/0.4 branch
RUN git clone https://github.com/pytorch/executorch.git \
    && cd executorch \
    # Update and pull submodules
    && git submodule sync \
    && git submodule update --init


# Copy and execute the installation script for dependencies
WORKDIR /workspace/executorch
RUN chmod +x ./install_requirements.sh \
    && ./install_requirements.sh

RUN git config --global user.email "dnb1654rrts@gmail.com"
RUN git config --global user.name "dinusha94"

WORKDIR /workspace/executorch/examples/arm/
RUN ./setup.sh --i-agree-to-the-contained-eula

# Set an entrypoint (optional, if you have specific actions for the container to perform)
CMD ["/bin/bash"]
