#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xcsv.hpp>
#include <nlohmann/json.hpp> // Include the nlohmann/json library
#include "src/objects/model.cpp"
#include "src/objects/gradient_descent.cpp"
#include "src/objects/autoencoder.cpp"
#include "src/utils.cpp"

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

    // Send error message
    if (argc < 2)
    {
        cerr << "Error: Missing config.json file path.\n";
        print_help(argv[0]);
        return 1;
    }

    // Parsed the config_file argument
    string config_file_path = argv[1];

    // Read the JSON file
    nlohmann::json config = read_json(config_file_path);

    // Parse the arguments from the JSON file
    string dataset_path = config["dataset_path"];
    vector<int> network_architecture = config["network_architecture"];
    unsigned int epochs = config["epochs"];
    int batch_size = config["batch_size"];
    float learning_rate = config["learning_rate"];

    // Load dataset
    ifstream infile(dataset_path);
    xt::xarray<double> dataset = xt::load_csv<double>(infile);
    infile.close();

    // Split x and y
    // Autoencoding now
    // TODO: make easy to switch between modes
    xt::xarray<double> x = xt::view(dataset, xt::all(), xt::range(0, dataset.shape(1))); // - 1
    xt::xarray<double> &y = x;
    // xt::xarray<double> y = xt::view(dataset, xt::all(), dataset.shape(1) - 1);

    // Scale data
    scale_data(x);
    scale_data(y);

    // Load model
    Autoencoder nn(network_architecture, x.shape(1));
    // Args are network architecture and input size of the data 

    // Load pretrained weights if desired
    if (config["pretrained_model_path"].is_null()) {
        // Do nothing, randomly initialized weights will stay still
    } else {
        // Load pretrained weights and loss 
        string pretrained_model_path = config["pretrained_model_path"];
        nn.load_weights(pretrained_model_path);
        nn.load_loss(pretrained_model_path);
    }

    // Train model
    nn.fit(x, epochs, batch_size, learning_rate);
    
    // Save trained weights and loss if desired
    if (!config["model_save_path"].is_null()) {
        string model_save_path = config["model_save_path"];
        nn.save_weights(model_save_path);
        nn.save_loss(model_save_path);
    }

    // Predict test set and save result if desired
    if (!config["pred_save_path"].is_null()) {
        // Predict
        xt:xarray<double> y_pred = nn.predict(x);
        // Save
        string pred_save_path = config["pred_save_path"];
        std::ofstream out_file (pred_save_path);
        xt::dump_csv(out_file, y_pred);
    }
    return 0;
}