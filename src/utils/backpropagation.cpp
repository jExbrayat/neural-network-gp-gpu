#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <tuple>
#include <cmath>
#include <vector>
#include <src/definition.hpp>
using namespace xt::placeholders; // to enable _ syntax
using namespace std;
using namespace xt;

std::tuple<std::vector<xarray<double>>, std::vector<xarray<double>>, xarray<double>>
make_gradient_descent(
    xarray<double> x_train,
    xarray<double> y_train, // shape must be (n, 1)
    int epochs,
    float learning_rate,
    std::vector<int> neurons_per_layer) // list of neurons in each layer
{

    // Define constants
    int dataset_size = x_train.shape()[0];
    int input_size = x_train.shape()[1];

    // Initialize network
    int num_layers = neurons_per_layer.size();
    std::vector<xarray<double>> weights(num_layers);
    std::vector<xarray<double>> biases(num_layers);

    // Initialize weights and biases for each layer
    weights[0] = xt::random::randn<double>({neurons_per_layer[0], input_size});
    biases[0] = xt::random::randn<double>({neurons_per_layer[0], 1});

    for (int l = 1; l < num_layers; l++)
    {
        weights[l] = xt::random::randn<double>({neurons_per_layer[l], neurons_per_layer[l - 1]});
        biases[l] = xt::random::randn<double>({neurons_per_layer[l], 1});
    }

    // Init mse array
    xarray<double> mse_array = xt::empty<double>({0});

    // Make gradient descent
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        float mse = 0;
        for (int i = 0; i < dataset_size; i++)
        {

            // Input layer
            xarray<double> a0 = xt::view(x_train, i, xt::all());
            a0 = a0.reshape({input_size, 1});
            std::vector<xarray<double>> activations(num_layers + 1);
            std::vector<xarray<double>> z_values(num_layers);

            activations[0] = a0;

            // Forward propagation
            for (int l = 0; l < num_layers; l++)
            {
                z_values[l] = xt::linalg::dot(weights[l], activations[l]) + biases[l];
                activations[l + 1] = sigma(z_values[l]);
            }

            // Output layer (prediction)
            xarray<double> a_final = activations[num_layers]; // the output after last layer

            // Compute MSE
            mse += std::pow(a_final(0, 0) - y_train(i, 0), 2) / dataset_size;

            // Backpropagation
            std::vector<xarray<double>> deltas(num_layers);
            deltas[num_layers - 1] = (a_final(0, 0) - y_train(i, 0)) * sigma_derivative(z_values[num_layers - 1]);

            for (int l = num_layers - 2; l >= 0; l--)
            {
                deltas[l] = xt::linalg::dot(xt::transpose(weights[l + 1]), deltas[l + 1]) * sigma_derivative(z_values[l]);
            }

            // Update weights and biases
            for (int l = 0; l < num_layers; l++)
            {
                auto gradient_w = xt::linalg::dot(deltas[l], xt::transpose(activations[l]));
                auto gradient_b = deltas[l];
                weights[l] -= learning_rate * gradient_w;
                biases[l] -= learning_rate * gradient_b;
            }
        }

        mse_array = xt::concatenate(xtuple(mse_array, xarray<double>({mse})));
    }

    return std::make_tuple(weights, biases, mse_array);
}
