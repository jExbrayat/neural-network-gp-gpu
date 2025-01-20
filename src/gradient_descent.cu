#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <vector>
#include <string>
#include "gradient_descent.cuh"
#include "utils.hpp"
#include "cuda_utils.cuh"
#include "debugging_utils.hpp"
#include "cuda_operations.cuh"
#include "cuda_members.cuh"

using namespace std;
using namespace xt;

// Define constructor
// Just init class members
GradientDescent::GradientDescent(const xarray<float> &x_train, const xarray<float> &y_train, vector<xarray<float>> &weights, vector<xarray<float>> &biases, const int batch_size) : x_train(x_train), y_train(y_train), weights(weights), biases(biases), batch_size(batch_size), XT2Cuda(x_train, y_train, weights, biases, batch_size) {
    num_layers = weights.size(); 

}


// Write gradient descent methods

void GradientDescent::forward_pass(float* x_ptr) {

    CudaMemberVectors& CMV = XT2Cuda.CudaMembers; 


    // Copy XBATCH_T into CMV.layer_activations[0] i.e. the network's input
    CudaMatrixMemory& network_input = CMV.layer_activations[0];
    // network_input.sendMatrix2Device(XBATCH_T.carray);

    CudaGrid Transpose;
    Transpose.setKernelGrid(16, 16, batch_size, x_train.shape(1));
    
    transposeKernel<<<Transpose.grid, Transpose.threads>>>(x_ptr, network_input.device_ptr, batch_size, x_train.shape(1));

    
    // Perform computations with cuda
    for (size_t l = 0; l < num_layers; l++) {
        CudaMatrixMemory& w = CMV.weights[l];
        
        CudaMatrixMemory& b = CMV.biases[l];

        CudaMatrixMemory& lo = CMV.layer_outputs[l];

        CudaMatrixMemory& la = CMV.layer_activations[l];

        CudaMatrixMemory& la_next = CMV.layer_activations[l + 1];
        
        CudaGrid matMulGrid;
        CudaGrid addGrid;
        CudaGrid sigmoidGrid;
        matMulGrid.setKernelGrid(16, 16, w.rows, la.cols);
        addGrid.setKernelGrid(16, 16, w.rows, la.cols);
        sigmoidGrid.setKernelGrid(16, 16, la_next.rows, la_next.cols);

        matrixMulKernel<<<matMulGrid.grid, matMulGrid.threads>>>(w.device_ptr, la.device_ptr, lo.device_ptr, w.rows, w.cols, la.cols); // w * la, write the result in lo
        addBiasToMatrixKernel<<<addGrid.grid, addGrid.threads>>>(lo.device_ptr, b.device_ptr, lo.device_ptr, lo.rows, lo.cols);
        sigmoidKernel<<<sigmoidGrid.grid, sigmoidGrid.threads>>>(lo.device_ptr, la_next.device_ptr, la_next.rows, la_next.cols);

    }

}

void GradientDescent::backward_pass(float* y_ptr, CudaMatrixMemory &yT_ptr, const int &current_batch_size, const float &learning_rate) {
    
    // Compute delta vectors
    // Transpose y_batch    
    int ybatch_rows = current_batch_size;
    int ybatch_cols = y_train.shape(1); // Nb of features

    CudaGrid Transpose;
    Transpose.setKernelGrid(16, 16, ybatch_rows, ybatch_cols);

    transposeKernel<<<Transpose.grid, Transpose.threads>>>(y_ptr, yT_ptr.device_ptr, ybatch_rows, ybatch_cols);
    

    // Create reference for easier reading
    CudaMemberVectors &CMV = XT2Cuda.CudaMembers;
    CudaMatrixMemory &delta = CMV.deltas[0];
    CudaMatrixMemory &lo = CMV.layer_outputs[num_layers - 1];
    CudaMatrixMemory &la_next = CMV.layer_activations[num_layers];

    // Init delta vectors

    // Create grid for cuda
    CudaGrid InitDelta;
    InitDelta.setKernelGrid(16, 16, delta.rows, delta.cols);
    
    // Perform soustraction. Store resutl in delta.device_ptr
    addMatrixToMatrix<<<InitDelta.grid, InitDelta.threads>>>(la_next.device_ptr, yT_ptr.device_ptr, -1.f, delta.device_ptr, delta.rows, delta.cols);
 
    // Store result in lo which is not a problem because this data can be altered
    sigmoidDerivativeKernel<<<InitDelta.grid, InitDelta.threads>>>(lo.device_ptr, lo.device_ptr, lo.rows, lo.cols);
 
    // Store the final result in delta.device_ptr
    matMulElementWise<<<InitDelta.grid, InitDelta.threads>>>(delta.device_ptr, lo.device_ptr, delta.device_ptr, delta.rows, delta.cols);


    // Compute remaining delta vectors
    for (size_t l = 1; l < num_layers; l++) {

        // Create references for easier reading
        CudaMatrixMemory &delta = CMV.deltas[l];
        CudaMatrixMemory &delta_previous = CMV.deltas[l - 1];
        CudaMatrixMemory &w = CMV.weights[num_layers - l];
        CudaMatrixMemory &lo = CMV.layer_outputs[num_layers - 1 - l]; 
        CudaMatrixMemory &wg = CMV.grad_weights[num_layers -l];
        
        // Create grids for cuda
        CudaGrid Transpose;
        CudaGrid ComputeDelta;
        Transpose.setKernelGrid(16, 16, w.rows, w.cols);        
        ComputeDelta.setKernelGrid(16, 16, delta.rows, delta.cols);

        // Compute transpose weights matrix. Store the result in wg because it is allocated already and can be altered without consequence
        transposeKernel<<<Transpose.grid, Transpose.threads>>>(w.device_ptr, wg.device_ptr, w.rows, w.cols);

        // Compute the dot product. Store the result in delta.device_ptr
        matrixMulKernel<<<ComputeDelta.grid, ComputeDelta.threads>>>(wg.device_ptr, delta_previous.device_ptr, delta.device_ptr, w.cols, w.rows, delta_previous.cols);
        
        // Compute sigmoid derivative. Store result in lo without consequence since its values will be recomputed in forward_pass
        sigmoidDerivativeKernel<<<ComputeDelta.grid, ComputeDelta.threads>>>(lo.device_ptr, lo.device_ptr, lo.rows, lo.cols);
        
        // Compute element wise multiplication. Store final result in delta 
        matMulElementWise<<<ComputeDelta.grid, ComputeDelta.threads>>>(delta.device_ptr, lo.device_ptr, delta.device_ptr, delta.rows, delta.cols);

    }


    // // Update weights and biases
    // for (int l = 0; l < num_layers; l++) {
    //     xarray<float> gradient_w = xt::linalg::dot(deltas[l], xt::transpose(layer_activations[l])) / current_batch_size; // Batch size may vary, at the end of epoch
    //     xarray<float> gradient_b = xt::mean(deltas[l], {1});
    //     gradient_b = gradient_b.reshape({gradient_b.size(), 1});

    //     weights[l] -= learning_rate * gradient_w;
    //     biases[l] -= learning_rate * gradient_b;
    // }

    for (size_t l = 0; l < num_layers; l++) {
        CudaMatrixMemory &w_grad = CMV.grad_weights[l];
        CudaMatrixMemory &b_grad = CMV.grad_biases[l];
        CudaMatrixMemory &w = CMV.weights[l];
        CudaMatrixMemory &b = CMV.biases[l];
        CudaMatrixMemory &la = CMV.layer_activations[l];
        CudaMatrixMemory &delta = CMV.deltas[num_layers - 1 - l]; // Reverse indexing

        CudaGrid GradientWGrid;
        GradientWGrid.setKernelGrid(16, 16, w_grad.rows, w_grad.cols);
        CudaGrid GradientBGrid;
        GradientBGrid.setKernelGrid(16, 16, b_grad.rows, b_grad.cols);
        CudaGrid TransposeGrid;
        TransposeGrid.setKernelGrid(16, 16, la.rows, la.cols);

        // Compute transpose of la. Store the result in la.device_ptr which has no consequence since it will be overwritten in next forward_pass
        transposeKernel<<<TransposeGrid.grid, TransposeGrid.threads>>>(la.device_ptr, la.device_ptr, la.rows, la.cols);
        
        // Compute dot product. Store result in w_grad
        matrixMulKernel<<<GradientWGrid.grid, GradientWGrid.threads>>>(delta.device_ptr, la.device_ptr, w_grad.device_ptr, delta.rows, delta.cols, la.rows);

        // Compute multiplication of matrix by scalar. Store result in w_grad
        // Multiply by learning rate at the same time
        float w_grad_scalar = learning_rate / current_batch_size;
        matrixScalarKernel<<<GradientWGrid.grid, GradientWGrid.threads>>>(w_grad.device_ptr, w_grad.device_ptr, w_grad_scalar, w_grad.rows, w_grad.cols);

        // Compute the mean of delta vector which gives the gradient of b. Store result in b_grad
        computeMeanKernel<<<GradientBGrid.grid, GradientBGrid.threads>>>(delta.device_ptr, b_grad.device_ptr, delta.rows, delta.cols);
        
        // Multiply by learning rate
        const float &b_grad_scalar = learning_rate;
        matrixScalarKernel<<<GradientBGrid.grid, GradientBGrid.threads>>>(b_grad.device_ptr, b_grad.device_ptr, b_grad_scalar, b_grad.rows, b_grad.cols);

        // Substract gradient to weights and biases
        addMatrixToMatrix<<<GradientWGrid.grid, GradientWGrid.threads>>>(w.device_ptr, w_grad.device_ptr, -1.f, w.device_ptr, w.rows, w.cols);
        addMatrixToMatrix<<<GradientBGrid.grid, GradientBGrid.threads>>>(b.device_ptr, b_grad.device_ptr, -1.f, b.device_ptr, b.rows, b.cols);

    }
}

void GradientDescent::train(const unsigned int &epochs, const float &learning_rate) {
    int dataset_size = x_train.shape()[0];
    int batch_number = (dataset_size / batch_size);

    int ybatch_rows = batch_size;
    int ybatch_cols = y_train.shape(1); // Nb of features

    // Allocate host memory for storing the MSE
    int MSE_memsize = sizeof(float) * ybatch_rows * 1;
    float *MSE = new float[MSE_memsize];

    CudaMatrixMemory yT_ptr(ybatch_cols, ybatch_rows); 

    for (unsigned int epoch = 0; epoch < epochs; epoch++) {

        cout << "Epoch: " << epoch << endl;
        float epoch_mse = 0;
        int batch_id = 0;

        for (int batch_start = 0; batch_start < dataset_size; batch_start += batch_size) {
            if (batch_start + batch_size > x_train.shape(0)) { // Fixed batch size. If the current batch exceeds the end of the dataset, break.
                break;
            }
            
            // Plot currently processed batch number and increment
            cout << "   Batch: " << batch_id << " / " << batch_number << endl;
            batch_id++;

            // Compute the current batch size, as defined normally but smaller if at the end of epoch
            int current_batch_size = batch_size;
            xarray<float> x_batch = xt::view(x_train, range(batch_start, batch_start + current_batch_size), all());
            xarray<float> y_batch = xt::view(y_train, range(batch_start, batch_start + current_batch_size), all());
            
            int cols = x_train.shape(1);
            float *x_ptr = XT2Cuda.x.device_ptr + cols * batch_start;
            // float* res = new float[batch_size * cols];
            // int ressize = batch_size * cols * sizeof(float);
            // cudaMemcpy(res, x_ptr, ressize, cudaMemcpyDeviceToHost);
            // print_carray(res, cols, batch_size, "print res");

            // Perform the forward pass
            forward_pass(x_ptr); // Modify the layer_activations and layer_outputs
            


            int ycols = y_train.shape(1);
            float *y_ptr = XT2Cuda.y.device_ptr + ycols * batch_start;
            // Perform the backward pass
            backward_pass(y_ptr, yT_ptr, current_batch_size, learning_rate); // Modify the weights and biases
 

            if (epoch % 10 == 0) {
                CudaMemberVectors &CMV = XT2Cuda.CudaMembers;
                CudaMatrixMemory &last_la = CMV.layer_activations[num_layers];
                // float* host_last_activation = CMV.layer_activations[num_layers].allocAndSend2Host();             
                // ArrayHandler last_activation;
                // last_activation.cast_carray(host_last_activation, CMV.layer_activations[num_layers].rows, CMV.layer_activations[num_layers].cols);
                // xarray<float> &last_activation = layer_activations[num_layers];

                // Compute the loss for the current batch
                CudaGrid Transpose;
                Transpose.setKernelGrid(16,16, batch_size, ycols);
                CudaGrid TransposeBack;
                TransposeBack.setKernelGrid(16, 16, ycols, batch_size);
                CudaGrid PowerTwo;
                PowerTwo.setKernelGrid(16, 16, ycols, batch_size);
                CudaGrid &Mean =PowerTwo;
                CudaGrid &Substract = PowerTwo;

                transposeKernel<<<Transpose.grid, Transpose.threads>>>(y_ptr, yT_ptr.device_ptr, batch_size, ycols);
                addMatrixToMatrix<<<Substract.grid, Substract.threads>>>(last_la.device_ptr, yT_ptr.device_ptr, -1.f, last_la.device_ptr, last_la.rows, last_la.cols);
                matrixPowerTwo<<<PowerTwo.grid, PowerTwo.threads>>>(last_la.device_ptr, last_la.device_ptr, last_la.rows, last_la.cols);
                transposeKernel<<<TransposeBack.grid, TransposeBack.threads>>>(last_la.device_ptr, last_la.device_ptr, ycols, batch_size);
                // Compute mean accross observations. Store the result in la which has no consequence since it will be overwritten in next forward pass
                computeMeanKernel<<<Mean.grid, Mean.threads>>>(last_la.device_ptr, last_la.device_ptr, batch_size, ycols);


                cudaMemcpy(MSE, last_la.device_ptr, MSE_memsize, cudaMemcpyDeviceToHost);
                std::cout << "Performed cudaMemcpy" << std::endl; 

                // xarray<float> squared_error = xt::pow(last_activation.xtarray - xt::transpose(y_batch), 2); // Error for each pixel of each observation
                // xarray<float> observation_mse = xt::mean(squared_error, {0}); // Mean over all the pixels in the observations
                ArrayHandler observation_mse;
                observation_mse.cast_carray(MSE, 1, batch_size);

                epoch_mse += xt::sum(observation_mse.xtarray)() / dataset_size;
            }
        }
        if (epoch % 10 == 0) {
            cout << "   MSE: " << epoch_mse << endl;
            loss_history.push_back(epoch_mse);

        }
    }
    delete[] MSE;
}


void GradientDescent::get_weights() {
    

    for (size_t l=0; l < num_layers; l++) {

        CudaMatrixMemory &DeviceLayerWeights = XT2Cuda.CudaMembers.weights[l];
        float *host_weights = DeviceLayerWeights.allocAndSend2Host();


        ArrayHandler GetWeights;
        GetWeights.cast_carray(host_weights, DeviceLayerWeights.rows, DeviceLayerWeights.cols);
        this->weights[l] = GetWeights.xtarray; // Perform a deep copy
        
        delete[] host_weights;
    }



}


void GradientDescent::get_biases() {
    

    for (size_t l=0; l < num_layers; l++) {

        CudaMatrixMemory &DeviceLayerBiases = XT2Cuda.CudaMembers.biases[l];
        float *host_biases = DeviceLayerBiases.allocAndSend2Host();


        ArrayHandler GetBiases;
        GetBiases.cast_carray(host_biases, DeviceLayerBiases.rows, DeviceLayerBiases.cols);
        this->biases[l] = GetBiases.xtarray; // Perform a deep copy

        delete[] host_biases;
    
    }


}

