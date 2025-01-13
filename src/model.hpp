#ifndef MODEL_HPP
#define MODEL_HPP

#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xrandom.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <optional>
#include "gradient_descent.cuh"

using namespace std;
using namespace xt;

// Define class structure
class Model
{
public:
    // Constructor
    Model(const vector<int> &architecture, const int &input_size);

    // Methods
    void load_weights(const string &path);
    void load_loss(const string &path);
    void save_weights(const string &path) const;
    void save_loss(const string &path) const;
    xarray<double> predict(const xarray<double> &x_test) const;

protected:
    vector<int> architecture; // Define the neural network architecture
    int input_size; // Define the input size of the data to fit predict
    vector<xarray<double>> weights; // Define vector of tensors for making operations on weights
    vector<xarray<double>> biases; // Define vector of tensors for making operations on biases
    vector<double> loss_history; // Store the loss over epochs

    void initialize_weights(); // Initialize weights. Used if no weights loaded
};

#endif
