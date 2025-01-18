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
GradientDescent::GradientDescent(const xarray<float> &x_train, const xarray<float> &y_train, vector<xarray<float>> &weights, vector<xarray<float>> &biases, const int batch_size) : x_train(x_train), y_train(y_train), weights(weights), biases(biases), batch_size(batch_size) {
    num_layers = weights.size(); 
    layer_outputs.resize(num_layers);
    layer_activations.resize(num_layers + 1);

    CudaMemberVectors& CMV = CudaMembers;
    CMV.biases.reserve(num_layers);
    CMV.deltas.reserve(num_layers);
    CMV.layer_activations.reserve(num_layers + 1);
    CMV.layer_outputs.reserve(num_layers);
    CMV.weights.reserve(num_layers);
    CMV.grad_biases.reserve(num_layers);
    CMV.grad_weights.reserve(num_layers);

    // Initialize cuda arrays (allocate memory)
    
    // Init first layer input, which is the transpose of x_batch
    // Note that the indexing of LA (layer_activations) is somehow décalé: LA_l is the input of the layer L and output of the layer l-1
    int larows = x_train.shape(1);
    int lacols = batch_size;
    CMV.layer_activations.emplace_back(larows, lacols);
    
    // Init first delta i.e. the delta tensor of the last layer
    int init_deltarows = x_train.shape(1); // nb features
    int init_deltacols = batch_size;
    CMV.deltas.emplace_back(init_deltarows, init_deltacols);

    for (size_t l = 0; l < num_layers; l++) {
        // Weights
        int wrows = weights[l].shape(0);
        int wcols = weights[l].shape(1);
        CMV.weights.emplace_back(wrows, wcols);

        // Weights gradients
        CMV.grad_weights.emplace_back(wrows, wcols);

        // Biases
        int brows = biases[l].shape(0);
        int bcols = biases[l].shape(1);
        CMV.biases.emplace_back(brows, bcols);

        // Biases gradients
        CMV.grad_biases.emplace_back(brows, bcols);

        // Layer output = W_l * LA_l + B_l
        int lorows = wrows;
        int locols = CMV.layer_activations[l].cols;
        CMV.layer_outputs.emplace_back(lorows, locols);

        // Layer activation = sigmoid( LO_{l-1} )
        // We are pushing the element l + 1 of the vector now (because of the initialization before the loop)
        CMV.layer_activations.emplace_back(lorows, locols);

        if (l > 0) { // Otherwise do nothing since the first value is initialized already
            int deltarows = weights[num_layers - l].shape(1);
            int deltacols = CMV.deltas[l - 1].cols;
            CMV.deltas.emplace_back(deltarows, deltacols);
        }


        cudaDeviceSynchronize();
    }  
}


// Write gradient descent methods

void GradientDescent::forward_pass(const xarray<float> &x_batch) {

    layer_activations[0] = xt::transpose(x_batch);
    
    // Transform xtarray into carray
    ArrayHandler XBATCH_T;
    XBATCH_T.cast_xtarray(layer_activations[0]);

    CudaMemberVectors& CMV = CudaMembers; 

    // Copy XBATCH_T into CMV.layer_activations[0] i.e. the network's input
    CudaMatrixMemory& network_input = CMV.layer_activations[0];
    network_input.sendMatrix2Device(XBATCH_T.carray);
    
    // Perform computations with cuda
    for (size_t l = 0; l < num_layers; l++) {
        CudaMatrixMemory& w = CMV.weights[l];
        ArrayHandler get_weights;
        get_weights.cast_xtarray(weights[l]);
        w.sendMatrix2Device(get_weights.carray);
        
        CudaMatrixMemory& b = CMV.biases[l];
        ArrayHandler get_biases;
        get_biases.cast_xtarray(biases[l]);
        b.sendMatrix2Device(get_biases.carray);

        CudaMatrixMemory& lo = CMV.layer_outputs[l];
        ArrayHandler get_lo;
        get_lo.cast_xtarray(layer_outputs[l]);
        lo.sendMatrix2Device(get_lo.carray);

        CudaMatrixMemory& la = CMV.layer_activations[l];
        ArrayHandler get_la;
        get_la.cast_xtarray(layer_activations[l]);
        la.sendMatrix2Device(get_la.carray);

        CudaMatrixMemory& la_next = CMV.layer_activations[l + 1];
        ArrayHandler get_la_next;
        get_la_next.cast_xtarray(layer_activations[l + 1]);
        la_next.sendMatrix2Device(get_la_next.carray);
        
        CudaGrid matMulGrid;
        CudaGrid addGrid;
        CudaGrid sigmoidGrid;
        matMulGrid.setKernelGrid(16, 16, w.rows, la.cols);
        addGrid.setKernelGrid(16, 16, w.rows, la.cols);
        sigmoidGrid.setKernelGrid(16, 16, la_next.rows, la_next.cols);

        matrixMulKernel<<<matMulGrid.grid, matMulGrid.threads>>>(w.device_ptr, la.device_ptr, lo.device_ptr, w.rows, w.cols, la.cols); // w * la, write the result in lo
        addBiasToMatrixKernel<<<addGrid.grid, addGrid.threads>>>(lo.device_ptr, b.device_ptr, lo.device_ptr, lo.rows, lo.cols);
        sigmoidKernel<<<sigmoidGrid.grid, sigmoidGrid.threads>>>(lo.device_ptr, la_next.device_ptr, la_next.rows, la_next.cols);

        // Perform computation on CPU
        xarray<float> CPU_lo = xt::linalg::dot(weights[l], layer_activations[l]) + biases[l];
        xarray<float> CPU_la_next = sigmoid(CPU_lo); 
        // Check computation
        checkCudaComputation(la_next, CPU_la_next, 0.01, "Check computation of LA NEXT of l = " + to_string(l));

        // Copy back the computations into the base pipeline
        float* w_host = w.allocAndSend2Host();
        float* b_host = b.allocAndSend2Host();
        float* lo_host = lo.allocAndSend2Host();
        float* la_host = la.allocAndSend2Host();
        float* la_next_host = la_next.allocAndSend2Host();

        // Assign to base pipeline
        ArrayHandler lo_xt;
        lo_xt.cast_carray(lo_host, lo.rows, lo.cols);
        layer_outputs[l] = lo_xt.xtarray;
        
        ArrayHandler la_next_xt;
        la_next_xt.cast_carray(la_next_host, la_next.rows, la_next.cols);
        layer_activations[l + 1] = la_next_xt.xtarray;  


        delete[] w_host;
        delete[] b_host;
        delete[] lo_host;
        delete[] la_host;
        delete[] la_next_host;
    }

}

void GradientDescent::backward_pass(const xarray<float> &y_batch, const int &current_batch_size, const float &learning_rate) {
    
    vector<xarray<float>> deltas(num_layers);

    // Init delta vector corresponding to the last layer
    xarray<float> &last_activation = layer_activations[num_layers];
    deltas[num_layers - 1] = (last_activation - xt::transpose(y_batch)) * sigmoid_derivative(layer_outputs[num_layers - 1]);

    for (int l = num_layers - 2; l >= 0; l--) {
        deltas[l] = xt::linalg::dot(xt::transpose(weights[l + 1]), deltas[l + 1]) * sigmoid_derivative(layer_outputs[l]);
    }

    // Compute delta vectors
        
    // Transpose y_batch and send to device
    xarray<float> refxt_ybatchT = xt::transpose(y_batch);
    ArrayHandler y_batchT;
    y_batchT.cast_xtarray(refxt_ybatchT);
    
    CudaMatrixMemory cmm_y_batchT(y_batch.shape(1), y_batch.shape(0)); 
    cmm_y_batchT.sendMatrix2Device(y_batchT.carray);
    
    checkCudaComputation(cmm_y_batchT, refxt_ybatchT, 0.001, "Check transposition of y_batch: ");

    // Create reference for easier reading
    CudaMemberVectors &CMV = CudaMembers;
    CudaMatrixMemory &delta = CMV.deltas[0];
    CudaMatrixMemory &lo = CMV.layer_outputs[num_layers - 1];
    CudaMatrixMemory &la_next = CMV.layer_activations[num_layers];

    // Init delta vectors

    // Create grid for cuda
    CudaGrid InitDelta;
    InitDelta.setKernelGrid(16, 16, delta.rows, delta.cols);
    
    // Perform soustraction. Store resutl in delta.device_ptr
    addMatrixToMatrix<<<InitDelta.grid, InitDelta.threads>>>(la_next.device_ptr, cmm_y_batchT.device_ptr, -1.f, delta.device_ptr, delta.rows, delta.cols);
 
    // Store result in lo which is not a problem because this data can be altered
    sigmoidDerivativeKernel<<<InitDelta.grid, InitDelta.threads>>>(lo.device_ptr, lo.device_ptr, lo.rows, lo.cols);
 
    // Store the final result in delta.device_ptr
    matMulElementWise<<<InitDelta.grid, InitDelta.threads>>>(delta.device_ptr, lo.device_ptr, delta.device_ptr, delta.rows, delta.cols);

    checkCudaComputation(delta, deltas[num_layers - 1], 0.001, "Check computation of the init of deltas: ");


    // // Compute remaining delta vectors
    // for (size_t l = 1; l < num_layers; l++) {

    //     // Create references for easier reading
    //     CudaMatrixMemory &delta = CMV.deltas[l];
    //     CudaMatrixMemory &delta_previous = CMV.deltas[l - 1];
    //     CudaMatrixMemory &w = CMV.weights[num_layers - l];
    //     CudaMatrixMemory &lo = CMV.layer_outputs[num_layers - 1 - l]; 
    //     CudaMatrixMemory &wg = CMV.grad_weights[num_layers -l];
        
    //     // Create grids for cuda
    //     CudaGrid Transpose;
    //     CudaGrid ComputeDelta;
    //     Transpose.setKernelGrid(16, 16, w.rows, w.cols);        
    //     ComputeDelta.setKernelGrid(16, 16, delta.rows, delta.cols);

    //     // Compute transpose weights matrix. Store the result in wg because it is allocated already and can be altered without consequence
    //     transposeKernel<<<Transpose.grid, Transpose.threads>>>(w.device_ptr, wg.device_ptr, w.rows, w.cols);

    //     // Compute the dot product
    //     matrixMulKernel<<<ComputeDelta.grid, ComputeDelta.threads>>>(resT, delta_previous.device_ptr, delta.device_ptr, w.cols, w.rows, delta_previous.cols);
        
    //     sigmoidDerivativeKernel<<<ComputeDelta.grid, ComputeDelta.threads>>>(lo.device_ptr, lo.device_ptr, lo.rows, lo.cols);
        
    //     matMulElementWise<<<ComputeDelta.grid, ComputeDelta.threads>>>(delta.device_ptr, lo.device_ptr, delta.device_ptr, delta.rows, delta.cols);
    // }

    // Update weights and biases
    for (int l = 0; l < num_layers; l++) {
        xarray<float> gradient_w = xt::linalg::dot(deltas[l], xt::transpose(layer_activations[l])) / current_batch_size; // Batch size may vary, at the end of epoch
        xarray<float> gradient_b = xt::mean(deltas[l], {1});
        gradient_b = gradient_b.reshape({gradient_b.size(), 1});

        weights[l] -= learning_rate * gradient_w;
        biases[l] -= learning_rate * gradient_b;
    }

    // for (size_t l = 0; l < num_layers; l++) {
    //     CudaMatrixMemory &w_grad = CMV.grad_weights[l];
    //     CudaMatrixMemory &b_grad = CMV.b_gradients[l];
    //     CudaMatrixMemory &w = CMV.weights[l];
    //     CudaMatrixMemory &b = CMV.biases[l];
    //     CudaMatrixMemory &la = CMV.layer_activations[l];
    //     CudaMatrixMemory &delta = CMV.deltas[l];

    //     CudaGrid GradientWGrid;
    //     GradientWGrid.setKernelGrid(16, 16, w_grad.rows, w_grad.rows);
    //     CudaGrid GradientBGrid;
    //     GradientBGrid.setKernelGrid(16, 16, b_grad.rows, b_grad.cols);
    //     CudaGrid TransposeGrid;
    //     TransposeGrid.setKernelGrid(16, 16, la.rows, la.cols);

    //     transposeKernel<<<TransposeGrid.grid, TransposeGrid.threads>>>(la.device_ptr, la.device_ptr, la.rows, la.cols);
    //     float *&laT_ptr = la.device_ptr;
        
    //     matrixMulKernel<<<GradientWGrid.grid, GradientWGrid.threads>>>(delta.device_ptr, laT_ptr, w_grad.device_ptr, delta.rows, delta.cols, la.rows); // B_cols is la.rows since laT_ptr is transpose of la
        
    //     float w_grad_scalar = learning_rate / current_batch_size;
    //     matrixScalarKernel<<<GradientWGrid.grid, GradientWGrid.threads>>>(w_grad.device_ptr, w_grad.device_ptr, w_grad_scalar, w_grad.rows, w_grad.cols);

    //     computeMeanKernel<<<GradientBGrid.grid, GradientBGrid.threads>>>(delta.device_ptr, b_grad.device_ptr, delta.rows, delta.cols);

    //     const float &b_grad_scalar = learning_rate;
    //     matrixScalarKernel<<<GradientBGrid.grid, GradientBGrid.threads>>>(b_grad.device_ptr, b_grad.device_ptr, b_grad_scalar, b_grad.rows, b_grad.cols);

    //     addMatrixToMatrix<<<GradientWGrid.grid, GradientWGrid.threads>>>(w_grad.device_ptr, w.device_ptr, -1.f, w.device_ptr, w.rows, w.cols);
    //     addMatrixToMatrix<<<GradientBGrid.grid, GradientBGrid.threads>>>(b_grad.device_ptr, b.device_ptr, -1.f, b.device_ptr, b.rows, b.cols);

    //     float *host_b = b.allocAndSend2Host();
    //     float *host_w = w.allocAndSend2Host();
    //     ArrayHandler xt_b;
    //     xt_b.cast_carray(host_b, b.rows, b.cols);
    //     ArrayHandler xt_w;
    //     xt_w.cast_carray(host_w, w.rows, w.cols);

    //     weights[l] = xt_w.xtarray;
    //     biases[l] = xt_b.xtarray;
    // }
}

void GradientDescent::train(const unsigned int &epochs, const float &learning_rate) {
    int dataset_size = x_train.shape()[0];
    int batch_number = (dataset_size / batch_size);

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

            // Perform the forward pass
            forward_pass(x_batch); // Modify the layer_activations and layer_outputs
            xarray<float> &last_activation = layer_activations[num_layers];

            // Perform the backward pass
            backward_pass(y_batch,  current_batch_size, learning_rate); // Modify the weights and biases
 
            // Compute the loss for the current batch
            xarray<float> squared_error = xt::pow(last_activation - xt::transpose(y_batch), 2); // Error for each pixel of each observation
            xarray<float> observation_mse = xt::mean(squared_error, {0}); // Mean over all the pixels in the observations
            epoch_mse += xt::sum(observation_mse)() / dataset_size;
        }
        cout << "   MSE: " << epoch_mse << endl;
        loss_history.push_back(epoch_mse);
    }
}

