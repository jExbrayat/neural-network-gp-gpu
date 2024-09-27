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

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        float mse = 0;
        for (int i = 0; i < dataset_size; i++)
        {

            // Input layer
            xarray<double> a0 = xt::view(x_train, i, xt::all());
            a0 = a0.reshape({2, 1});

            // First hidden layer
            auto z1 = xt::linalg::dot(w1, a0);
            auto a1 = sigma(z1);

            // Second hidden layer
            auto z2 = xt::linalg::dot(w2, a1) + b2;
            auto a2 = sigma(z2);

            // Third hidden layer
            auto z3 = xt::linalg::dot(w3, a2) + b3;
            auto a3 = sigma(z3); // prediction, shape (1, 1)

            // Compute MSE
            mse += std::pow(a3(1, 1) - y_train(i, 1), 2);

            // Make backpropagation

            xarray<double> delta3 = (a3 - y_train(i, 1)) * sigma_derivative(z3);
            xarray<double> delta2 = linalg::dot(transpose(w3), delta3) * sigma_derivative(z2);
            xarray<double> delta1 = linalg::dot(transpose(w2), delta2) * sigma_derivative(z1);

            xarray<double> gradient_b1 = delta1;
            xarray<double> gradient_b2 = delta2;
            xarray<double> gradient_b3 = delta3;

            xarray<double> gradient_w1 = zeros<double>(w1.shape());
            xarray<double> gradient_w2 = zeros<double>(w2.shape());
            xarray<double> gradient_w3 = zeros<double>(w3.shape());

            for (int j = 0; j < w1.shape()[0]; j++) {
                for (int k = 0; k < w1.shape()[1]; k++) {
                    gradient_w1[j, k] = a0[k] * delta1[j];
                }
            }

            for (int j = 0; j < w2.shape()[0]; j++) {
                for (int k = 0; k < w2.shape()[1]; k++) {
                    gradient_w2[j, k] = a1[k] * delta2[j];
                }
            }

            for (int j = 0; j < w3.shape()[0]; j++) {
                for (int k = 0; k < w3.shape()[1]; k++) {
                    gradient_w3[j, k] = a2[k] * delta3[j];
                }
            }


            // Updating biases and weights
            b1 -= learning_rate * gradient_b1;
            b2 -= learning_rate * gradient_b2; 
            b3 -= learning_rate * gradient_b3; 
            w1 -= learning_rate * gradient_w1;  
            w2 -= learning_rate * gradient_w2;  
            w3 -= learning_rate * gradient_w3;  

        }
    }

    return std::make_tuple(w1, w2, w3, b1, b2, b3);
}