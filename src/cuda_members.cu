
#include "cuda_utils.cuh"
#include "cuda_members.cuh"
#include "utils.hpp"
#include <iostream>
#include <xtensor/xarray.hpp>  // For xt::xarray
#include <xtensor/xview.hpp>   // Optional: For views if used
#include <xtensor/xio.hpp>     // Optional: For I/O support
#include <vector>              // For std::vector
using namespace std;

Xtensor2CudaMatrixMemory::Xtensor2CudaMatrixMemory(const xt::xarray<float>&x_train, const xt::xarray<float>&y_train,  vector<xt::xarray<float>>& weights, vector<xt::xarray<float>>& biases,const  int batch_size) : x(x_train.shape(0), x_train.shape(1)), y(y_train.shape(0), y_train.shape(1)) {
    int num_layers = weights.size(); 


    cout << "Allocating X and Y now" << endl;
    cudaDeviceSynchronize();
    ArrayHandler castx;
    castx.cast_xtarray(x_train);
    x.sendMatrix2Device(castx.carray);

    ArrayHandler casty;
    casty.cast_xtarray(y_train);
    y.sendMatrix2Device(casty.carray);

    
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
        ArrayHandler wtest;
        wtest.cast_xtarray(weights[l]);
        CMV.weights.emplace_back(wrows, wcols);
        CMV.weights[l].sendMatrix2Device(wtest.carray);

        // Weights gradients
        CMV.grad_weights.emplace_back(wrows, wcols);

        // Biases
        int brows = biases[l].shape(0);
        int bcols = biases[l].shape(1);
        ArrayHandler btest;
        btest.cast_xtarray(biases[l]);
        CMV.biases.emplace_back(brows, bcols);
        CMV.biases[l].sendMatrix2Device(btest.carray);

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