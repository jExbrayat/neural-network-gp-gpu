
# Parallelized Feedforward Neural Network with CUDA

This project provides a framework for training a feedforward neural network with CUDA-based parallelization. It includes CLI-based functionality for model configuration, training, prediction, and output visualization.

## Overview

The program uses a json configuration file to specify the model's parameters and saving path.

## Usage

Run the program with the following command:

```bash
build/main path/to/config.json
```

### Configuration File Format

The configuration file should be in JSON format with the following parameters:

- **`"dataset_path"`** *string*: Path to the training dataset.
- **`"epochs"`** *int*: Number of training epochs.
- **`"train_test_split"`** *float*: Proportion of the dataset to be dedicated for training e.g. 0.8.
- **`"batch_size"`** *int*: Size of each batch for training.
- **`"learning_rate"`** *float*: Learning rate for the optimizer.
- **`"network_architecture"`** *list<int>*: List representing the number of neurons per layer. For example, `[4, 8, 1]` specifies an architecture with two hidden layers and a single output neuron.
- **`"pretrained_model_path"`** *string | null*: Path to a pretrained model to load initial weights and loss values, allowing continuation of training.
- **`"model_save_path"`** *string | null*: Path to save the trained model weights and loss data. *Note:* Ensure the directory exists before running the program, as it does not currently create missing directories.
- **`"pred_save_path"`** *string*: Path to save predictions for the test set.

## User Guide

GnuPlot is required for plotting purposes, such as MNIST autoencoder input/output images comparisons and model loss tracking.  
If GnuPlot is not installed, the program will still run but without the ability to generate plots.

## Developer Guide

### Compiling the Project

#### With project-included libraries

The program is thought to be compiled without Docker in case of admin rights restrictions.

1) `cd` in the project's root directory.  
2) Build the program by including the header only libraries:
```bash
g++ -std=c++17 -I "libraries/include" -I "." src/main.cpp -o build/main
```

#### With Docker
The compilation process is streamlined using Docker. Ensure you have Docker Engine installed.  
If not, follow this [installation guide](https://docs.docker.com/engine/install/ubuntu/).

1) `cd` in the project's root directory.  
2) Build the container with the Dockerfile:
```bash
sudo docker build -t cuda_nn_project .
```
After this step, an image  named *cuda_nn_project* of an Ubuntu version with all necessary packages and g++ compiler is installed. 

3) Run the Docker image:
```bash
sudo docker run -it --rm -v $(pwd):/usr/src/app cuda_nn_project
```
This command opens a terminal within the container, where you can execute and test the program.  
The `-v` argument tells the container to read the files in the host's project root directory (pwd) when being in `container:/usr/src/app` (see docker volumes).  
The container files will be cleaned up directly after exiting it due to the `--rm` argument.

4) Compile the project in the running container:  
```bash
g++ -I "." -I "/root/.local/share/mamba/include/" src/main.cpp -o build/main
```

## Data structure
### MNIST dataset
mnist_train: vector<60_000>(vector<784>(uint8 0-255))  
mnist_test: vector<60_000>(int 0-9)
