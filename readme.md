
# Parallelized Feedforward Neural Network with CUDA

This project provides a framework for training a feedforward neural network with CUDA-based parallelization. It includes CLI-based functionality for model configuration, training, prediction, and output visualization.

## Overview

The program uses a json configuration file to specify the model's parameters and saving paths.

## Usage
- Go in the project's root directory.  
- Compile.
```bash
make build/main_program
```
- Create your configuration file `config.json` in the project's root directory.

- Run.
```bash
make run
```

### Configuration File Format

The configuration file should be in JSON format with the following parameters:

- **`"dataset_path"`** *string*: Path to the training dataset.
- **`"epochs"`** *int*: Number of training epochs.
- **`"train_test_split"`** *float*: Proportion of the dataset to be dedicated for training e.g. 0.8.
- **`"batch_size"`** *int*: Size of each batch for training.
- **`"learning_rate"`** *float*: Learning rate for the optimizer.
- **`"network_architecture"`** *list<int>*: List representing the number of neurons per layer. For example, `[4, 8, 1]` specifies an architecture with two hidden layers and a single output neuron.
- **`"pretrained_model_path"`** *string | null*: Folder path to a pretrained model to load initial weights and loss values, allowing continuation of training.
- **`"model_save_path"`** *string | null*: Folder path to save the trained model weights and loss data. *Note:* Ensure the directory exists before running the program, as it does not currently create missing directories.
- **`"pred_save_path"`** *string*: File path to save predictions for the test set.

### Example

Train in seconds an autoencoder for denoising sinusoidal signals with the following configuration:

```json
{
    "dataset_path": "datasets/noisy_sinus.csv",
    "train_test_split": 0.8,
    "network_architecture": [8, 4, 8, 20],
    "epochs": 100,
    "batch_size": 64,
    "learning_rate": 0.1,
    "model_save_path": null,
    "pretrained_model_path":null,
    "pred_save_path": null 
}
```
