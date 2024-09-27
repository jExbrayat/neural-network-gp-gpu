#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp> 
#include <xtensor/xview.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xcsv.hpp>
using namespace xt::placeholders;  // to enable _ syntax
using namespace std;
using namespace xt;
#include "definition.hpp"
#include "backpropagation.cpp"

void print_help(const std::string& program_name) {
    std::cout << "Usage: " << program_name << " [options] <input_file.csv>\n"
            << "Options:\n"
            << "  --help                 Show this help message and exit\n"
            << "  <input_file.csv>       Path to the input CSV file containing data\n\n"
            << "Example:\n"
            << "  " << program_name << " input_data.csv\n";
}

int main(int argc, char* argv[])
{
    // Check if the --help argument is passed
    if (argc > 1 && std::string(argv[1]) == "--help") {
        print_help(argv[0]);
        return 0;
    }

    // Open the file passed as argument
    std::string file_name = argv[1];
    std::ifstream file(file_name);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << file_name << std::endl;
        return 1;
    }

    // Use xtensor-io to load the CSV data into an xtensor xarray
    xt::xarray<double> dataset = load_csv<double>(file, ',');
    
    // Shuffle
    shuffleArray(dataset); 


    // Split into train and test sets
    int train_size = abs(0.8 * dataset.shape(0));
    
    xt::xarray<double> x_train = xt::view(dataset, xt::range(_, train_size), xt::range(0, 2));
    xt::xarray<double> y_train = xt::view(dataset, xt::range(_, train_size), xt::range(2, 3));
    
    xt::xarray<double> x_test = xt::view(dataset, xt::range(train_size, _), xt::range(0, 2));
    xt::xarray<double> y_test = xt::view(dataset, xt::range(train_size, _), xt::range(2, 3)); // shape (n, 1)

    std::tuple weights_biases = make_gradient_descent(x_train, y_train, 100, 0.1);

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

    gnuplot(x_test, y_test_pred, "Dataset coloured according to the predicted cluster");
    gnuplot(x_test, true_pred, "Dataset coloured according to correctness of prediction");

    return 0;
}