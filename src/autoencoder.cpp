#include "autoencoder.hpp"
#include <iostream>
#include "mnist_reader.hpp"
#include "utils.hpp"
#include "gradient_descent.cuh"

using namespace std;

void Autoencoder::fit(const xarray<double> &x_train, const unsigned int &epochs, const int &batch_size, const float &learning_rate)
{
    // Create an instance of GradientDescent
    GradientDescent gradientDescent(x_train, x_train, weights, biases, batch_size);
    
    // Train the model using the train method of GradientDescent
    gradientDescent.train(epochs, learning_rate);
    
    // Retrieve the results
    weights = gradientDescent.weights;
    biases = gradientDescent.biases;
    loss_history.insert(loss_history.end(), gradientDescent.loss_history.begin(), gradientDescent.loss_history.end()); // Append the updated loss history
}

tuple<xarray<double>, xarray<double>, xarray<double>, xarray<double>> Autoencoder::load_mnist_dataset(const float &train_test_split) {
        // For debugging/testing purpose:
        // Define number of train images and test images to load according to the train test split proportion defined by the user
        int train_limit = round(train_test_split * 60000); // 60000 is the max number of train images to load
        int test_limit = round(train_test_split * 10000); // 10000 is the max number of test images to load
        mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("datasets/autoencoding/mnist", train_limit, test_limit);
        xarray<double> x_train = transform_mnist_images(dataset.training_images, {dataset.training_images.size(), 784}); // shape (N, 784)
        xarray<double> y_train = transform_mnist_labels(dataset.training_labels, {dataset.training_labels.size(), 1}); // shape (N, 1)
        xarray<double> x_test = transform_mnist_images(dataset.test_images, {dataset.test_images.size(), 784});
        xarray<double> y_test = transform_mnist_labels(dataset.test_labels, {dataset.test_labels.size(), 1});
    
    return std::make_tuple(x_train, y_train, x_test, y_test);
}
