#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp> 
#include <xtensor/xview.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>
using namespace xt::placeholders;  // to enable _ syntax
using namespace std;
using namespace xt;
#include "definition.hpp"
#include "backpropagation.cpp"


int main()
{

    // Create two random datasets with different caracteristics
    auto x1 = create_random_dataset(0, 1.4, 500); // shape (n, 2)
    xt::xarray<int> y1 = xt::ones<int>({500, 1}); // shape (n, 1)
    xt::xarray<double> dataset1 = xt::concatenate(xt::xtuple(x1, y1), 1); // shape (n, 3)

    auto x2 = create_random_dataset(5, 0.8, 500);
    xt::xarray<int> y2 = xt::zeros<int>({500, 1});
    xt::xarray<double> dataset2 = xt::concatenate(xt::xtuple(x2, y2), 1);

    // Concatenate the two datasets and shuffle them
    xt::xarray<double> dataset = xt::concatenate(xtuple(dataset1, dataset2), 0);

    // Shuffle
    shuffleArray(dataset); 

    // Split into train and test sets
    xt::xarray<double> x_train = xt::view(dataset, xt::range(_, 800), xt::range(0, 2));
    xt::xarray<double> y_train = xt::view(dataset, xt::range(_, 800), xt::range(2, 3));
    
    xt::xarray<double> x_test = xt::view(dataset, xt::range(800, _), xt::range(0, 2));
    xt::xarray<double> y_test = xt::view(dataset, xt::range(800, _), xt::range(2, 3)); // shape (n, 1)

    std::tuple weights_biases = make_gradient_descent(x_train, y_train, 10, 0.1);

    auto [w1, w2, w3, b1, b2, b3] = weights_biases;

    // Predict probas
    xarray<double> y_test_proba = zeros<double>({0, 1});
    for (int i = 0; i < y_test.size(); i++) {

        // Input layer
        xarray<double> a0 = xt::view(x_test, i, xt::all());
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

        // Append to the prediction vector
        y_test_proba = concatenate(xt::xtuple(y_test_proba, a3), 0); // shape (n, 1)
    }

    // Convert proba to class prediction
    xt::xarray<int> y_test_pred = xt::empty<int>(y_test_proba.shape()); // shape (n, 1)
    for (int i = 0; i < y_test_proba.shape()[0]; i++) {
        if (y_test_proba(i, 0) <= 0.5) {
            y_test_pred(i, 0) = 0;
        } else {
            y_test_pred(i, 0) = 1;
        }
    }

    // Compute vector taking 1 if prediction is correct
    xarray<int> true_pred = empty<int>(y_test.shape()); // shape (n, 1)
    for (int i = 0; i < y_test.size(); i++) {
        true_pred(i, 1) = (y_test(i, 1) == y_test_pred(i, 1)) ? 1 : 0; // Assign 1 or 0 based on the condition
    } 

    cout << "\nPrecision:\n";
    double precision = std::accumulate(true_pred.begin(), true_pred.end(), 0.0) / y_test.size();
    cout << precision << endl;

    gnuplot(x_test, y_test_pred);
    gnuplot(x_test, true_pred);

    return 0;
}