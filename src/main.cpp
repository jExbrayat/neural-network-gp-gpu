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
#include "src/utils/mnist_reader.hpp"
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
    std::optional<std::string> pretrained_model_path = 
        (config["pretrained_model_path"] != "") ? std::optional<std::string>(config["pretrained_model_path"]) : std::nullopt;
    std::string model_save_path = config["model_save_path"];

    // Use xtensor-io to load the CSV data into an xtensor xarray
    xt::xarray<double> x_train;
    xt::xarray<double> y_train;
    xt::xarray<double> x_test;
    xt::xarray<double> y_test;
    if (dataset_path == "mnist") {

        auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("datasets/mnist_data");    
        x_train = transform_mnist_images(dataset.training_images, {dataset.training_images.size(), 784}); // shape (N, 784)
        y_train = transform_mnist_labels(dataset.training_labels, {dataset.training_labels.size(), 1}); // shape (N, 1)
        x_test = transform_mnist_images(dataset.test_images, {dataset.test_images.size(), 784});
        y_test = transform_mnist_labels(dataset.test_labels, {dataset.test_labels.size(), 1});
    
    } else {
        std::ifstream dataset_file(dataset_path);
        if (!dataset_file.is_open())
        {
            std::cerr << "Error: Could not open file " << dataset_path << std::endl;
            return 1;
        }
        xt::xarray<double> dataset = load_csv<double>(dataset_file, ',');
    }

    int input_csv_cols = x_train.shape(1);
    int x_dataset_cols = input_csv_cols;

    // Shuffle
    // TODO: does it work ?
    // shuffleArray(dataset);

    // Take subdataset
    x_train = xt::view(x_train, xt::range(0, 100), xt::all());
    
    // Scale images to [0; 1]
    x_train = scale_data(x_train);
    x_test = scale_data(x_test);
    
    // Find good weights
    std::tuple weights_biases = make_gradient_descent(x_train, x_train, epochs, learning_rate, network_architecture, pretrained_model_path);
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
    gnuplot_image_plot(a_plotted, "autoencoded img");
    cout << endl << " prediction shape " << a_plotted.shape(0) << "," << a.shape(1);
    std::cout << "\nRMSE:\n";
    std::cout << sqrt(mse_array(mse_array.size() - 1)) << endl;

    dump_model(weights, biases, mse_array, model_save_path);

    return 0;
}