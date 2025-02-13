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
#include "cuda_members.cuh"

using namespace std;
using namespace xt;

class GradientDescent
{
public:
    // Constructor
    GradientDescent(const xarray<float>& x_train, const xarray<float>& y_train, 
                    vector<xarray<float>>& weights, vector<xarray<float>>& biases, const int batch_size);
    
    // Method to start training
    void train(const unsigned int& epochs, const float& learning_rate);

    // Class members
    vector<xarray<float>> weights;   // Weights of the network
    vector<xarray<float>> biases;    // Biases of the network
    vector<float> loss_history;      // History of loss over epochs
    const xarray<float> x_train;           // Training data (inputs)
    const xarray<float> y_train;           // Labels corresponding to the training data
    const int batch_size;

    void get_weights();
    void get_biases();
private:
    // Forward pass through the network
    void forward_pass(float* x_ptr);

    // Backward pass to calculate gradients and update weights/biases
    void backward_pass(float* y_ptr,  CudaMatrixMemory &yT_ptr, const int& current_batch_size, const float& learning_rate);

    // Class members for layer outputs and activations
    int num_layers;                   // Number of layers in the network

    Xtensor2CudaMatrixMemory XT2Cuda;
};

#endif // GRADIENT_DESCENT_CUH
