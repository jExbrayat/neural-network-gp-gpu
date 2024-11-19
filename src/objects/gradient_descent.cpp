#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <optional>
#include <src/definition.hpp>

using namespace std;
using namespace xt;

class GradientDescent
{
public:
    // Constructor
    GradientDescent(const xarray<double> &x_train, const xarray<double> &y_train, vector<xarray<double>> &weights, vector<xarray<double>> &biases);
    
    // Method
    void train(const unsigned int &epochs, const unsigned int &batch_size, const float &learning_rate);
    
protected:
    vector<xarray<double>> weights; // Define vector of tensors for making operations on weights
    vector<xarray<double>> biases;  // Define vector of tensors for making operations on biases
    vector<double> loss_history; // Store the loss over epochs
    xarray<double> x_train;
    xarray<double> y_train;
    int num_layers;

private:
    // Method
    xarray<double> forward_pass(const xarray<double> &x_batch);
};

// Define constructor
// Just init class members
GradientDescent::GradientDescent(const xarray<double> &x_train, const xarray<double> &y_train, vector<xarray<double>> &weights, vector<xarray<double>> &biases) : x_train(x_train), y_train(y_train), weights(weights), biases(biases) {
    num_layers = weights.size(); 
}


// Write gradient descent methods

xarray<double> GradientDescent::forward_pass(const xarray<double> &x_batch) {

    std::vector<xarray<double>> layer_outputs(num_layers);
    std::vector<xarray<double>> layer_activations(num_layers + 1);
    layer_activations[0] = xt::transpose(x_batch);

    for (int l = 0; l < num_layers; l++) {
        layer_outputs[l] = xt::linalg::dot(weights[l], layer_activations[l]) + biases[l];
        layer_activations[l + 1] = sigma(layer_outputs[l]); // sigma is defined in src/utils/utils.cpp
    }

    xarray<double>& last_activation = layer_activations[num_layers];
    return last_activation;
}

