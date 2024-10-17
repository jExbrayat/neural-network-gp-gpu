#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xcsv.hpp>
#include <nlohmann/json.hpp> // Include the nlohmann/json library
#include "src/definition.hpp"
#include "src/utils/utils.cpp"
#include "src/utils/backpropagation.cpp"
using namespace xt::placeholders; // to enable _ syntax
using namespace std;
using namespace xt;
using json = nlohmann::json;

void print_help(const std::string &program_name)
{
    std::cout << "Usage: " << program_name << " config.json\n";
    std::cout << "Pass a config.json file containing parameters.\n";
}

int main(int argc, char *argv[])
{
    // Check if the --help argument is passed
    if (argc > 1 && std::string(argv[1]) == "--help")
    {
        print_help(argv[0]);
        return 0;
    }

    if (argc < 2)
    {
        std::cerr << "Error: Missing config.json file path.\n";
        print_help(argv[0]);
        return 1;
    }

    std::string config_file = argv[1];

    // Read the JSON file
    std::ifstream file(config_file);
    if (!file.is_open())
    {
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
    if (!dataset_file.is_open())
    {
        std::cerr << "Error: Could not open file " << dataset_path << std::endl;
        return 1;
    }
    xt::xarray<double> dataset = load_csv<double>(dataset_file, ',');
    int input_csv_cols = dataset.shape(1);
    int x_dataset_cols = input_csv_cols;

    // Shuffle
    shuffleArray(dataset);

    // Split into train and test sets
    int train_size = abs(0.8 * dataset.shape(0));

    xt::xarray<double> x_train = xt::view(dataset, xt::range(_, train_size), xt::all());
    xt::xarray<double> y_train = x_train;

    xt::xarray<double> x_test = xt::view(dataset, xt::range(train_size, _), xt::all());
    xt::xarray<double> y_test = x_test;
    // shape (n, k)

    // Find good weights
    std::tuple weights_biases = make_gradient_descent(x_train, y_train, epochs, learning_rate, network_architecture);
    auto [weights, biases, mse_array] = weights_biases;


    // Predict one output to check result
    int num_layers = weights.size(); // number of layers in the network
    int i = 0; // Observation id

    // Set input layer
    xarray<double> a = xt::view(x_test, i, xt::all());
    a = a.reshape({x_dataset_cols, 1}); // Reshape to match input dimension

    // Forward propagation through all layers
    for (int l = 0; l < num_layers; l++)
    {
        auto z = xt::linalg::dot(weights[l], a) + biases[l];
        a = sigma(z); // Apply the activation function to each layer output
    }
    // a is the predicted value for the target value y_test[i]

    auto a_plotted = a.reshape({a.size()});
    gnuplot_loss_plot(mse_array, "Loss");
    gnuplot_loss_plot(a_plotted, "Autoencoded series");
    cout << endl << a_plotted.shape(0) << "," << a.shape(1);
    cout << endl << mse_array.shape(0) << "," << mse_array.shape(1);
    std::cout << "\nRMSE:\n";
    std::cout << sqrt(mse_array(mse_array.size() - 1)) << endl;

    return 0;
}