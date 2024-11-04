#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <tuple>
#include <cmath>
#include <vector>
#include <optional>
#include <src/definition.hpp>
#include <string>
using namespace xt::placeholders; // to enable _ syntax
using namespace std;
using namespace xt;

void print_shapes(xarray<double> array, string name) {
    cout << name << std::endl;
    cout << array.shape(0) << ", " << array.shape(1) << std::endl;
}

std::tuple<std::vector<xt::xarray<double>>, std::vector<xt::xarray<double>>, xt::xarray<double>>
make_gradient_descent(
    xt::xarray<double> x_train, // shape (n, k_in)
    xt::xarray<double> y_train, // shape (n, k_out)
    int epochs,
    int batch_size,
    float learning_rate,
    std::vector<int> neurons_per_layer, // list of neurons in each layer
    std::optional<std::string> pretrained_model_path = std::nullopt // optional path
) {
    int dataset_size = x_train.shape()[0];
    int input_size = x_train.shape()[1];
    int num_layers = neurons_per_layer.size();

    std::vector<xt::xarray<double>> weights(num_layers);
    std::vector<xt::xarray<double>> biases(num_layers);
    xt::xarray<double> mse_array;
    if (pretrained_model_path) {
        // Load pretrained weights, biases, and MSE array
        load_model(weights, biases, mse_array, *pretrained_model_path, num_layers);
    } else {
        // Initialize weights and biases
        weights[0] = xt::random::randn<double>({neurons_per_layer[0], input_size});
        biases[0] = xt::random::randn<double>({neurons_per_layer[0], 1});

        for (int l = 1; l < num_layers; l++) {
            weights[l] = xt::random::randn<double>({neurons_per_layer[l], neurons_per_layer[l - 1]});
            biases[l] = xt::random::randn<double>({neurons_per_layer[l], 1});
        }

        // Initialize mse_array as an empty array for new training
        mse_array = xt::empty<double>({0});
    }

    int batch_number = (dataset_size / batch_size);
    for (int epoch = 0; epoch < epochs; epoch++) {
        cout << "Epoch: " << epoch << endl;
        float epoch_mse = 0;

        int batch_id = 0; 
        for (int batch_start = 0; batch_start < dataset_size; batch_start += batch_size) {
            cout << "   Batch: " << batch_id << " / " << batch_number << endl;
            batch_id ++;

            int current_batch_size = std::min(batch_size, dataset_size - batch_start);
            auto x_batch = xt::view(x_train, range(batch_start, batch_start + current_batch_size), all());
            auto y_batch = xt::view(y_train, range(batch_start, batch_start + current_batch_size), all());

            std::vector<xarray<double>> activations(num_layers + 1);
            std::vector<xarray<double>> z_values(num_layers);
            activations[0] = xt::transpose(x_batch);
            // print_shapes(activations[0], "activations_0");

            // Forward propagation
            for (int l = 0; l < num_layers; l++) {
                z_values[l] = xt::linalg::dot(weights[l], activations[l]) + biases[l];
                // (w^n_0, w^n_1) dot (784, batch_size) + (w^n_0,) = (w^n_0, batch_size)
                activations[l + 1] = sigma(z_values[l]);
                // print_shapes(activations[l + 1], "activations_" + std::to_string(l+1));
            }

            xarray<double> a_final = activations[num_layers];
            xarray<double> squared_error = xt::pow(a_final - xt::transpose(y_batch), 2);
            xarray<double> mse_i = xt::mean(squared_error, {0});
            // Compute means accross the first axis (output shape: (cols,))
            // i.e. compute mean squared error for each observation in the batch
            // print_shapes(a_final, "a_final");
            // print_shapes(xt::transpose(y_batch), "y_batch_tranpose");
            // cout << mse_i.shape(0) << endl;
            // cout << mse_i << endl;
            epoch_mse += xt::sum(mse_i)() / dataset_size;

            // Backpropagation
            std::vector<xarray<double>> deltas(num_layers);
            deltas[num_layers - 1] = (a_final - xt::transpose(y_batch)) * sigma_derivative(z_values[num_layers - 1]);
            // print_shapes(deltas[num_layers - 1], "delta_" + std::to_string(num_layers - 1));
            
            for (int l = num_layers - 2; l >= 0; l--) {
                deltas[l] = xt::linalg::dot(xt::transpose(weights[l + 1]), deltas[l + 1]) * sigma_derivative(z_values[l]);
                // (w^n+1_1, w^n+1_0) dot (w^n+1_0, batch_size) * (w^n_0, batch_size) = (w^n_0, batch_size)
                // print_shapes(deltas[l], "delta_" + std::to_string(l));
            }

            // Accumulate gradients over the batch
            for (int l = 0; l < num_layers; l++) {
                xarray<double> gradient_w = xt::linalg::dot(deltas[l], xt::transpose(activations[l])) / current_batch_size;
                xarray<double> gradient_b = xt::mean(deltas[l], {1});
                gradient_b = gradient_b.reshape({gradient_b.size(), 1});
                // print_shapes(gradient_w, "gradient_w_" + to_string(l));
                // cout << "right gradient b" << endl;
                // cout << gradient_b.shape(0) << " " << gradient_b.size() << endl;
                // cout << gradient_b.dimension() << endl;
                weights[l] -= learning_rate * gradient_w;
                biases[l] -= learning_rate * gradient_b;
            }
        }
        mse_array = xt::concatenate(xtuple(mse_array, xarray<double>({epoch_mse})));
    }

    return std::make_tuple(weights, biases, mse_array);
}
