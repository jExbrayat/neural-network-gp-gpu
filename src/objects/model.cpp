#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xrandom.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <optional>
#include "src/include/gradient_descent.hpp"
#include "src/include/model.hpp"
#include "src/definition.hpp"

using namespace std;
using namespace xt;

// Define class constructor
Model::Model(const vector<int> &architecture, const int &input_size) : architecture(architecture), input_size(input_size)
{
    // Define constructor for Model: architecture on the left is the object's member, on the right
    // is the constructor argument
    initialize_weights();
}

// Write class functions

void Model::initialize_weights()
// Set weights biases with random normal distribution
{
    weights.push_back(xt::random::randn<double>({architecture[0], input_size}));
    biases.push_back(xt::random::randn<double>({architecture[0], 1}));

    for (size_t l = 1; l < architecture.size(); l++) {
        xt::xarray<double> w = xt::random::randn<double>({architecture[l], architecture[l - 1]});
        xt::xarray<double> b = xt::random::randn<double>({architecture[l], 1});
        weights.push_back(w);
        biases.push_back(b);
    }
}

void Model::load_weights(const string &path)
{
    // Implement loading weights from a file
}

void Model::load_loss(const string &path)
{
    // Implement loading loss from a file
}

void Model::save_weights(const string &path) const
{
    // Implement saving weights to a file
}

void Model::save_loss(const string &path) const
{
    // Implement saving loss to a file
}

xarray<double> Model::predict(const xarray<double> &x_test) const
{
    xarray<double> a = xt::transpose(x_test);
    for (size_t l = 0; l < weights.size(); ++l)
    {
        a = sigma(xt::linalg::dot(weights[l], a) + biases[l]);
    }
    return a;
}
