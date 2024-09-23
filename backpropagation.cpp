#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <tuple>
#include <cmath>
#include "utils.cpp"
using namespace xt::placeholders; // to enable _ syntax
using namespace std;
using namespace xt;

std::tuple<
    xt::xarray<double>,
    xt::xarray<double>,
    xt::xarray<double>,
    xt::xarray<double>,
    xt::xarray<double>,
    xt::xarray<double>
>
make_gradient_descent(
    xarray<double> x_train,
    xarray<double> y_train, // shape must be (1, n)
    int epochs,
    float learning_rate
) {

    // Define constants
    int dataset_size = x_train.shape()[0]; 


    // Initialize network

    // Weights
    xarray<double> w1 = xt::random::randn<double>({3, 2});
    xarray<double> w2 = xt::random::randn<double>({5, 3});
    xarray<double> w3 = xt::random::randn<double>({1, 5});

    // Biasses
    xarray<double> b1 = xt::random::randn<double>({3, 1});
    xarray<double> b2 = xt::random::randn<double>({5, 1});
    xarray<double> b3 = xt::random::randn<double>({1, 1});

    // Define gradient matrices
    xarray<double> nabla_w1 = zeros<double>(w1.shape());
    xarray<double> nabla_w2 = zeros<double>(w2.shape());
    xarray<double> nabla_w3 = zeros<double>(w3.shape());

    // Init predicted values array
    xarray<double> y_pred = zeros<double>({1, dataset_size});

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        float mse = 0;
        for (int i = 0; i < dataset_size; i++)
        {

            // Input layer
            xarray<double> a0 = xt::view(x_train, i, xt::all());
            a0 = a0.reshape({1, 2});

            // First hidden layer
            auto z1 = xt::linalg::dot(w1, a0);
            auto a1 = sigma(z1);

            // Second hidden layer
            auto z2 = xt::linalg::dot(w2, a1) + b2;
            auto a2 = sigma(z2);

            // Third hidden layer
            auto z3 = xt::linalg::dot(w3, a2) + b3;
            auto a3 = sigma(z3); // prediction

            // Compute MSE
            mse += std::pow(a3(1, 1) - y_train(1, i), 2);

            // Make backpropagation

            xarray<double> delta3 = 
        }
    }

    return std::make_tuple(w1, w2, w3, b1, b2, b3);
}