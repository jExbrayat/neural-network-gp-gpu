#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <vector>
#include <string>
#include "gradient_descent.cuh"
#include "utils.hpp"
#include "cuda_utils.cuh"

using namespace std;
using namespace xt;

// Define constructor
// Just init class members
GradientDescent::GradientDescent(const xarray<double> &x_train, const xarray<double> &y_train, vector<xarray<double>> &weights, vector<xarray<double>> &biases, const int batch_size) : x_train(x_train), y_train(y_train), weights(weights), biases(biases), batch_size(batch_size) {
    num_layers = weights.size(); 
    layer_outputs.resize(num_layers);
    layer_activations.resize(num_layers + 1);

    // Initialize cuda arrays (allocate memory)
    for (size_t l = 0; l < num_layers; l++) {
        // Weights
        int wrows = weights[l].shape(0);
        int wcols = weights[l].shape(1);
        CudaMatrixMemory LayerWeights(wrows, wcols);
        cuda_weights.push_back(LayerWeights);

        // Biases
        int brows = biases[l].shape(0);
        int bcols = biases[l].shape(1);
        CudaMatrixMemory LayerBiases(brows, bcols);
        cuda_biases.push_back(LayerBiases);
    }
}


// Write gradient descent methods

void GradientDescent::forward_pass(const xarray<double> &x_batch) {

    layer_activations[0] = xt::transpose(x_batch);
    
    // Transform xtarray into carray
    ArrayHandler XBATCH_T;
    XBATCH_T.cast_xtarray(layer_activations[0]);
    
    // // Allocate cuda memory
    // vector<CudaMatrixMemory> cuda_l_o;
    // vector<CudaMatrixMemory> cuda_l_a;
    
    // // Init the first layer activation
    // CudaMatrixMemory init_l_a(XBATCH_T.rows, XBATCH_T.cols);
    // init_l_a.sendMatrix2Device(XBATCH_T.carray);
    // cuda_l_a.push_back(init_l_a);

    // // Initialize cuBLAS
    // cublasHandle_t handle;
    // cublasCreate(&handle);
    
    // // Allocate iteratively
    // for (size_t l = 0; l < num_layers; l++) {
    //     // Cast xtarrays
    //     ArrayHandler WEIGHTS;
    //     ArrayHandler BIASES;
    //     WEIGHTS.cast_xtarray(weights[l]);
    //     BIASES.cast_xtarray(biases[l]);

    //     // Send matrices in cuda
    //     CudaMatrixMemory cuda_w(WEIGHTS.rows, WEIGHTS.cols);
    //     CudaMatrixMemory cuda_b(BIASES.rows, BIASES.cols);
    //     cuda_w.sendMatrix2Device(WEIGHTS.carray);
    //     cuda_b.sendMatrix2Device(BIASES.carray);

    //     // Perform operation with cuBLAS
    //     float alpha = 1.0f;
    //     float beta = 0.f;
    //     int M = WEIGHTS.rows;
    //     int K = WEIGHTS.cols;
    //     int N = cuda_l_a[l].cols;
    //     // Allocate memory for result
    //     CudaMatrixMemory result(M, N);
    //     cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, cuda_w.device_ptr, K, cuda_l_a[l].device_ptr, N, &beta, result.device_ptr, N);
    //     cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, &alpha, result.device_ptr, N, &beta, cuda_b.device_ptr, N, result.device_ptr, N);
    //     // Allocate memory for activated result
    //     CudaMatrixMemory activated_result(M, N);
    //     CudaKernel activateLayer;
    //     activateLayer.setKernelFunction(sigmoidKernel);
    //     activateLayer.setKernelGrid(16, 16, M, N);
    //     activateLayer.runKernel(result.device_ptr, activated_result.device_ptr, M, N);
                
    //     cuda_l_o.push_back(result);
    //     cuda_l_a.push_back(activated_result);
    // } 

    for (size_t l = 0; l < num_layers; l++) {
        layer_outputs[l] = xt::linalg::dot(weights[l], layer_activations[l]) + biases[l];
        layer_activations[l + 1] = sigmoid(layer_outputs[l]); // sigmoid is defined in utils/utils.cpp
    }
}

void GradientDescent::backward_pass(const xarray<double> &y_batch, const int &current_batch_size, const float &learning_rate) {
    
    vector<xarray<double>> deltas(num_layers);

    // Init delta vector corresponding to the last layer
    xarray<double> &last_activation = layer_activations[num_layers];
    deltas[num_layers - 1] = (last_activation - xt::transpose(y_batch)) * sigmoid_derivative(layer_outputs[num_layers - 1]);

    for (int l = num_layers - 2; l >= 0; l--) {
        deltas[l] = xt::linalg::dot(xt::transpose(weights[l + 1]), deltas[l + 1]) * sigmoid_derivative(layer_outputs[l]);
    }

    // Update weights and biases
    for (int l = 0; l < num_layers; l++) {
        xarray<double> gradient_w = xt::linalg::dot(deltas[l], xt::transpose(layer_activations[l])) / current_batch_size; // Batch size may vary, at the end of epoch
        xarray<double> gradient_b = xt::mean(deltas[l], {1});
        gradient_b = gradient_b.reshape({gradient_b.size(), 1});

        weights[l] -= learning_rate * gradient_w;
        biases[l] -= learning_rate * gradient_b;
    }
}

void GradientDescent::train(const unsigned int &epochs, const float &learning_rate) {
    int dataset_size = x_train.shape()[0];
    int batch_number = (dataset_size / batch_size);

    for (unsigned int epoch = 0; epoch < epochs; epoch++) {

        cout << "Epoch: " << epoch << endl;
        float epoch_mse = 0;
        int batch_id = 0;

        for (int batch_start = 0; batch_start < dataset_size; batch_start += batch_size) {
            // Plot currently processed batch number and increment
            cout << "   Batch: " << batch_id << " / " << batch_number << endl;
            batch_id++;

            // Compute the current batch size, as defined normally but smaller if at the end of epoch
            int current_batch_size = std::min(batch_size, dataset_size - batch_start);
            xarray<double> x_batch = xt::view(x_train, range(batch_start, batch_start + current_batch_size), all());
            xarray<double> y_batch = xt::view(y_train, range(batch_start, batch_start + current_batch_size), all());

            // Perform the forward pass
            forward_pass(x_batch); // Modify the layer_activations and layer_outputs
            xarray<double> &last_activation = layer_activations[num_layers];

            // Perform the backward pass
            backward_pass(y_batch,  current_batch_size, learning_rate); // Modify the weights and biases
 
            // Compute the loss for the current batch
            xarray<double> squared_error = xt::pow(last_activation - xt::transpose(y_batch), 2); // Error for each pixel of each observation
            xarray<double> observation_mse = xt::mean(squared_error, {0}); // Mean over all the pixels in the observations
            epoch_mse += xt::sum(observation_mse)() / dataset_size;
        }
        cout << "   MSE: " << epoch_mse << endl;
        loss_history.push_back(epoch_mse);
    }
}

