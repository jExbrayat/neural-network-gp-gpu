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
    xarray<double> x_train, // shape (n, k_in)
    xarray<double> y_train, // shape (n, k_out)
    int epochs,
    float learning_rate,
    std::vector<int> neurons_per_layer) // list of neurons in each layer
{

    // cout << "x_train";
    // cout << endl<< x_train.shape(0) << "," << x_train.shape(1) <<endl;
    // cout << "y_train";
    // cout << endl<< y_train.shape(0) << "," << y_train.shape(1) <<endl;


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
    // cout << "weights";
    // cout << endl<< weights[0].shape(0) << "," << weights[0].shape(1) <<endl;


    for (int l = 1; l < num_layers; l++)
    {
        weights[l] = xt::random::randn<double>({neurons_per_layer[l], neurons_per_layer[l - 1]});
        biases[l] = xt::random::randn<double>({neurons_per_layer[l], 1});
        // cout << endl<< weights[l].shape(0) << "," << weights[l].shape(1) <<endl;
    }


    // Init mse array
    xarray<double> mse_array = xt::empty<double>({0});

    // Make gradient descent
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        cout << endl << "Epoch: " << epoch << endl;

        float mse = 0;
        for (int i = 0; i < dataset_size; i++)
        {

            // Input layer
            xarray<double> a0 = xt::view(x_train, i, xt::all());
            a0 = a0.reshape({input_size, 1}); // Transpose vector, shape (k_in, 1)

            std::vector<xarray<double>> activations(num_layers + 1);
            std::vector<xarray<double>> z_values(num_layers);

            activations[0] = a0;
            // cout << "a0" << endl;
            // cout << activations[0].shape(0) << "," << activations[0].shape(1) <<endl;


            // Forward propagation
            for (int l = 0; l < num_layers; l++)
            {
                z_values[l] = xt::linalg::dot(weights[l], activations[l]) + biases[l];
                // (n0, k_in) dot (k_in, 1) + (n0, 1) = (n0, 1) for layer 0 
                activations[l + 1] = sigma(z_values[l]);
                // cout << "z and a" << endl;
                // cout << z_values[l].shape(0) << "," << z_values[l].shape(1) <<endl;
                // cout << endl<< activations[l+1].shape(0) << "," << activations[l+1].shape(1) <<endl;
            }

            // Output layer (prediction)
            xarray<double> a_final = activations[num_layers]; // the output at last layer
                                                                  // shape (k_out, 1)

            // cout << endl<< a_final.shape(0) << "," << a_final.shape(1) <<endl;

            // Retrieve and reshape target value
            xarray<double> y_train_i = xt::view(y_train, i, xt::all());
            y_train_i = y_train_i.reshape({y_train_i.size(), 1}); // Reshape the target for the error computation
                                                                // Shape (k_out, 1)

            // Compute MSE
            xarray<double> mse_i = xt::pow(a_final - y_train_i, 2) / input_size;
            mse += xt::sum(mse_i)() / dataset_size;

            // Backpropagation
            std::vector<xarray<double>> deltas(num_layers);
            deltas[num_layers - 1] = (a_final - y_train_i) * sigma_derivative(z_values[num_layers - 1]);
            // shape ((k_out, 1) - (k_out, 1)) * (k_out, 1)  = (k_out, 1)

            // cout << "deltas" << endl;
            // cout << deltas[num_layers -1].shape(0) << "," << deltas[num_layers -1].shape(1) <<endl;

            for (int l = num_layers - 2; l >= 0; l--)
            {
                // TODO: invert order in the dot product ?
                deltas[l] = xt::linalg::dot(xt::transpose(weights[l + 1]), deltas[l + 1]) * sigma_derivative(z_values[l]);
                // transpose(k_out, n_neurons) dot (k_out, 1) * (n_neurons, 1) = (n_neurons, 1)

                // cout << deltas[l].shape(0) << "," << deltas[l].shape(1) <<endl;
                
            }

            // Update weights and biases
            for (int l = 0; l < num_layers; l++)
            {
                // cout << deltas[l].shape(0) << "," << deltas[l].shape(1) <<endl;
                // cout << activations[l].shape(0) << "," << activations[l].shape(1) <<endl;
                // cout << endl;
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
