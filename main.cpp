#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp> 
#include <xtensor/xview.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>
using namespace xt::placeholders;  // to enable _ syntax
using namespace std;
using namespace xt;
#include "utils.cpp"
#include "backpropagation.cpp"


int main()
{

    // Create two random datasets with different caracteristics
    auto x1 = create_random_dataset(0, 1.4, 500);
    xt::xarray<int> y1 = xt::ones<int>({500, 1});
    xt::xarray<double> dataset1 = xt::concatenate(xt::xtuple(x1, y1), 1);

    auto x2 = create_random_dataset(5, 0.8, 500);
    xt::xarray<int> y2 = xt::zeros<int>({500, 1});
    xt::xarray<double> dataset2 = xt::concatenate(xt::xtuple(x2, y2), 1);

    // Concatenate the two datasets and shuffle them
    xt::xarray<double> dataset = xt::concatenate(xtuple(dataset1, dataset2), 0);

    // Shuffle
    shuffleArray(dataset); 

    // Split into train and test sets
    xt::xarray<double> x_train = xt::view(dataset, xt::range(_, 800), xt::range(0, 2));
    xt::xarray<double> y_train = xt::view(dataset, xt::range(_, 800), 2);
    
    xt::xarray<double> x_test = xt::view(dataset, xt::range(800, _), xt::range(0, 2));
    xt::xarray<double> y_test = xt::view(dataset, xt::range(800, _), 2);

    // Display a sample of the dataset in the console
    for (int i=0; i < 10; i++) {
        std::cout << xt::view(x_train, i, xt::range(0, 2))
        << xt::view(y_train, i)
        << std::endl;
    }

    gnu_plot(x_train);

    std::tuple weights_biases = make_gradient_descent(x_train, y_train, 10, 0.1);

    auto [w1, w2, w3, b1, b2, b3] = weights_biases;

    // Predict probas
    xarray<double> y_test_pred = zeros<double>({1, 1});
    for (int i = 0; i < y_test.size(); i++) {

        // Input layer
        xarray<double> a0 = xt::view(x_test, i, xt::all());
        a0 = a0.reshape({1, 2});

        // First hidden layer
        auto z1 = xt::linalg::dot(w1, a0);
        auto a1 = sigma(z1);

        // Second hidden layer
        auto z2 = xt::linalg::dot(w2, a1) + b2;
        auto a2 = sigma(z2);

        // Third hidden layer
        auto z3 = xt::linalg::dot(w3, a2) + b3;
        auto a3 = sigma(z3); // prediction, shape (1, 1)

        // Append to the prediction vector
        y_test_pred = concatenate(xt::xtuple(y_test_pred, a3), 0);
    }

    cout << y_test_pred;

    return 0;
}