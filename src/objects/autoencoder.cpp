#include "src/include/autoencoder.hpp"
#include "iostream"

using namespace std;

void Autoencoder::fit(const xarray<double> &x_train, const unsigned int &epochs, const int &batch_size, const float &learning_rate)
{
    // Create an instance of GradientDescent
    GradientDescent gradientDescent(x_train, x_train, weights, biases);
    
    // Train the model using the train method of GradientDescent
    gradientDescent.train(epochs, batch_size, learning_rate);
    
    // Retrieve the results
    loss_history = gradientDescent.loss_history;  // Store loss history from GradientDescent
    weights = gradientDescent.weights;
    biases = gradientDescent.biases;
}