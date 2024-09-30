#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp> 
#include <xtensor/xview.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xcsv.hpp>
#include "definition.hpp"
#include "backpropagation.cpp"
#include <nlohmann/json.hpp> // Include the nlohmann/json library
using namespace xt::placeholders;  // to enable _ syntax
using namespace std;
using namespace xt;
using json = nlohmann::json;


void print_help(const std::string& program_name) {
    std::cout << "Usage: " << program_name << " config.json\n";
    std::cout << "Pass a config.json file containing parameters.\n";
}

int main(int argc, char* argv[]) {
    // Check if the --help argument is passed
    if (argc > 1 && std::string(argv[1]) == "--help") {
        print_help(argv[0]);
        return 0;
    }

    if (argc < 2) {
        std::cerr << "Error: Missing config.json file path.\n";
        print_help(argv[0]);
        return 1;
    }

    std::string config_file = argv[1];
    
    // Read the JSON file
    std::ifstream file(config_file);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open config.json file.\n";
        return 1;
    }
    json config;
    file >> config;

    // Parse the arguments from the JSON file
    std::string dataset_path = config["dataset_path"];
    int epochs = config["epochs"];
    float learning_rate = config["learning_rate"];
    std::string prediction_mode = config["prediction_mode"];
    std::vector<int> network_architecture = config["network_architecture"];

    // Use xtensor-io to load the CSV data into an xtensor xarray
    std::ifstream dataset_file(dataset_path);
    if (!dataset_file.is_open()) {
        std::cerr << "Error: Could not open file " << dataset_path << std::endl;
        return 1;
    }
    xt::xarray<double> dataset = load_csv<double>(dataset_file, ',');
    int input_csv_cols = dataset.shape(1);
    int x_dataset_cols = input_csv_cols - 1;
    
    // Shuffle
    shuffleArray(dataset); 


    // Split into train and test sets
    int train_size = abs(0.8 * dataset.shape(0));

    xt::xarray<double> x_train = xt::view(dataset, xt::range(_, train_size), xt::range(0, input_csv_cols - 1));
    xt::xarray<double> y_train = xt::view(dataset, xt::range(_, train_size), xt::range(input_csv_cols - 1, input_csv_cols));
    
    xt::xarray<double> x_test = xt::view(dataset, xt::range(train_size, _), xt::range(0, input_csv_cols - 1));
    xt::xarray<double> y_test = xt::view(dataset, xt::range(train_size, _), xt::range(input_csv_cols - 1, input_csv_cols));; // shape (n, 1)

    std::tuple weights_biases = make_gradient_descent(x_train, y_train, epochs, learning_rate, network_architecture);

    auto [weights, biases, mse_array] = weights_biases;

    // Predict probabilities
    xarray<double> y_test_proba = xt::empty<double>(y_test.shape());
    int num_layers = weights.size(); // number of layers in the network

    for (int i = 0; i < y_test.shape(0); i++) {
        
        // Input layer
        xarray<double> a = xt::view(x_test, i, xt::all());
        a = a.reshape({x_dataset_cols, 1}); // Reshape to match input dimension

        // Forward propagation through all layers
        for (int l = 0; l < num_layers; l++) {
            auto z = xt::linalg::dot(weights[l], a) + biases[l];
            a = sigma(z);  // Apply the activation function to each layer output
        }

        // Append the final output (prediction) to the prediction vector
        y_test_proba(i, 0) = a(0, 0);  // shape (n, 1)
    }

    if (prediction_mode == "classification") {
        // Convert proba to class prediction
        xt::xarray<int> y_test_pred = empty<int>(y_test.shape()); // shape (n, 1)
        for (int i = 0; i < y_test.shape(0); i++) {
            if (y_test_proba(i, 0) <= 0.5) {
                y_test_pred(i, 0) = 0;
            } else {
                y_test_pred(i, 0) = 1;
            }
        }

        // Compute vector taking 1 if prediction is correct
        xarray<int> true_pred = empty<int>(y_test.shape()); // shape (n, 1)
        for (int i = 0; i < y_test.size(); i++) {
            true_pred(i, 0) = (y_test(i, 0) == y_test_pred(i, 0)) ? 1 : 0; // Assign 1 or 0 based on the condition
        } 

        double precision = std::accumulate(true_pred.begin(), true_pred.end(), 0.0) / y_test.shape(0);

        std::cout << "\nPrecision:\n";
        std::cout << precision << endl;
        if (x_dataset_cols == 2) { // If dataset is two-dimensional, plot
            gnuplot(x_test, y_test_pred, "Dataset coloured according to the predicted cluster");
            gnuplot(x_test, true_pred, "Dataset coloured according to correctness of prediction");
        }
    }
    
    if (prediction_mode == "regression") {
        gnuplot_ypred_ytrue(y_test, y_test_proba, "y pred vs y test");
    }

    gnuplot_loss_plot(mse_array, "Loss");
    std::cout << "\nRMSE:\n";
    std::cout << sqrt(mse_array(mse_array.size() - 1)) << endl;

    return 0;
}