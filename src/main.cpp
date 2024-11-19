#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xcsv.hpp>
#include <nlohmann/json.hpp> // Include the nlohmann/json library
#include "src/objects/model.cpp"
#include "src/objects/gradient_descent.cpp"
#include "src/utils/utils.cpp"
#include "src/definition.hpp"

using namespace std;
using namespace xt;

void print_help(const string &program_name)
{
    std::cout << "Usage: " << program_name << " config.json\n";
    std::cout << "Pass a config.json file containing parameters.\n";
}

int main(int argc, char *argv[]) {

    // Check if the --help argument is passed
    if (argc > 1 && string(argv[1]) == "--help")
    {
        print_help(argv[0]);
        return 0;
    }

    if (argc < 2)
    {
        cerr << "Error: Missing config.json file path.\n";
        print_help(argv[0]);
        return 1;
    }

    // Parsed the config_file argument
    string config_file = argv[1];

    // Read the JSON file
    std::ifstream file(config_file);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open config.json file.\n";
        return 1;
    }
    nlohmann::json config;
    file >> config;

    // Parse the arguments from the JSON file
    string dataset_path = config["dataset_path"];
    unsigned int epochs = config["epochs"];
    int batch_size = config["batch_size"];
    float learning_rate = config["learning_rate"];
    vector<int> network_architecture = config["network_architecture"];
    string model_save_path = config["model_save_path"];

    xt::xarray<double> dataset = load_xarray_from_csv(dataset_path);
    xt::xarray<double> x = xt::view(dataset, xt::all(), xt::range(0, dataset.shape(1) - 1));
    xt::xarray<double> y = xt::view(dataset, xt::all(), dataset.shape(1) - 1);

    Model nn(network_architecture, x.shape(1));

    nn.fit(x, y, epochs, batch_size, learning_rate);
    xt:xarray<double> y_pred = nn.predict(x);
    std::ofstream out_file ("models/junk/junk.csv");
    xt::dump_csv(out_file, y_pred);

    return 0;
}