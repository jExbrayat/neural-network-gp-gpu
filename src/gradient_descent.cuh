#ifndef GRADIENT_DESCENT_CUH
#define GRADIENT_DESCENT_CUH

#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <optional>
#include "cuda_utils.cuh"

using namespace std;
using namespace xt;

class GradientDescent
{
public:
    // Constructor
    GradientDescent(const xarray<double>& x_train, const xarray<double>& y_train, 
                    vector<xarray<double>>& weights, vector<xarray<double>>& biases, const int batch_size);
    
    // Method to start training
    void train(const unsigned int& epochs, const float& learning_rate);

    // Class members
    vector<xarray<double>> weights;   // Weights of the network
    vector<xarray<double>> biases;    // Biases of the network
    vector<double> loss_history;      // History of loss over epochs
    xarray<double> x_train;           // Training data (inputs)
    xarray<double> y_train;           // Labels corresponding to the training data
    const int batch_size;

private:
    // Forward pass through the network
    void forward_pass(const xarray<double>& x_batch);

    // Backward pass to calculate gradients and update weights/biases
    void backward_pass(const xarray<double>& y_batch, const int& current_batch_size, const float& learning_rate);

    // Class members for layer outputs and activations
    int num_layers;                   // Number of layers in the network
    vector<xarray<double>> layer_outputs;   // Layer outputs (linear activations)
    vector<xarray<double>> layer_activations;   // Layer activations after applying activation function

    // Class members for cuda
    vector<CudaMatrixMemory> cuda_weights; 
    vector<CudaMatrixMemory> cuda_biases;
    vector<CudaMatrixMemory> cuda_layer_outputs;  
    vector<CudaMatrixMemory> cuda_layer_activations; 
};

#endif // GRADIENT_DESCENT_CUH
