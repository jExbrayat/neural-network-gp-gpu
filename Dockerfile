# This Dockerfile builds an environment with the needed c++ dependencies for project compilation

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
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /usr bin/micromamba && \
# Download archive and extract micromamba binary in /usr/bin 
   eval "$(/usr/bin/micromamba shell hook -s posix)" && \
   micromamba install -c conda-forge -y xtensor-blas && \
   micromamba clean --all --yes

# Set the work directory to the app folder
WORKDIR /usr/src/app

# Build the project
# RUN g++ -I "." -I "/root/.local/share/mamba/include/" src/main.cpp -o build/main