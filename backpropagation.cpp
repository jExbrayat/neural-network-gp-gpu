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
    xt::xarray<double>,
    xt::xarray<double>>
make_gradient_descent(
    xarray<double> x_train,
    xarray<double> y_train, // shape must be (n, 1)
    int epochs,
    float learning_rate)
{

    // Define constants
    int dataset_size = x_train.shape()[0];

    // Initialize network

    // Weights
    xarray<double> w1 = xt::random::randn<double>({3, 2});
    xarray<double> w2 = xt::random::randn<double>({5, 3});
    xarray<double> w3 = xt::random::randn<double>({1, 5});

    // Biases
    xarray<double> b1 = xt::random::randn<double>({3, 1});
    xarray<double> b2 = xt::random::randn<double>({5, 1});
    xarray<double> b3 = xt::random::randn<double>({1, 1});

    // Init mse array
    xarray<double> mse_array = xt::empty<double>({0});
    
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        float mse = 0;
        for (int i = 0; i < dataset_size; i++)
        {

            // Input layer
            xarray<double> a0 = xt::view(x_train, i, xt::all());
            a0 = a0.reshape({2, 1});

            // First hidden layer
            auto z1 = xt::linalg::dot(w1, a0) + b1;
            auto a1 = sigma(z1);

            // Second hidden layer
            auto z2 = xt::linalg::dot(w2, a1) + b2;
            auto a2 = sigma(z2);

            // Third hidden layer
            auto z3 = xt::linalg::dot(w3, a2) + b3;
            auto a3 = sigma(z3); // prediction, shape (1, 1)

            // Compute MSE
            mse += std::pow(a3(0, 0) - y_train(i, 0), 2) / dataset_size;

            // Make backpropagation

            xarray<double> delta3 = (a3 - y_train(i, 0)) * sigma_derivative(z3);
            xarray<double> delta2 = linalg::dot(transpose(w3), delta3) * sigma_derivative(z2);
            xarray<double> delta1 = linalg::dot(transpose(w2), delta2) * sigma_derivative(z1);

            xarray<double> gradient_b1 = delta1;
            xarray<double> gradient_b2 = delta2;
            xarray<double> gradient_b3 = delta3;

            auto gradient_w1 = xt::linalg::dot(delta1, xt::transpose(a0));
            auto gradient_w2 = xt::linalg::dot(delta2, xt::transpose(a1));
            auto gradient_w3 = xt::linalg::dot(delta3, xt::transpose(a2));

            // Updating biases and weights
            b1 -= learning_rate * gradient_b1;
            b2 -= learning_rate * gradient_b2; 
            b3 -= learning_rate * gradient_b3; 
            w1 -= learning_rate * gradient_w1;  
            w2 -= learning_rate * gradient_w2;  
            w3 -= learning_rate * gradient_w3;  

        }
        mse_array = xt::concatenate(xtuple(mse_array, xarray<double>({mse})));
    }

    return std::make_tuple(w1, w2, w3, b1, b2, b3, mse_array);
}