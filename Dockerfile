# This Dockerfile builds image with project's source directory, c++ dependencies
# and g++ compiler.

# Use an Ubuntu base image
FROM ubuntu:latest

# Install g++ compiler, curl, and bzip2
RUN apt-get update && \
    apt-get install -y g++ curl bzip2 && \
    apt-get clean

# Install xtensor-dev
RUN apt-get install -y xtensor-dev && \
    apt-get clean

# Install micromamba for xtensor-blas
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba && \
# Download archive and extract bin/micromamba file in current directory
   eval "$(/bin/micromamba shell hook -s posix)" && \
   micromamba install -c -y conda-forge xtensor-blas && \
   micromamba clean --all --yes

# Set the work directory to the app folder
WORKDIR /usr/src/app


# Instead of the following, use docker volumes
# Copy the project files into the container
# COPY . .

# Build the project
# RUN g++ -I "." -I "/root/.local/share/mamba/include/" src/main.cpp -o build/main