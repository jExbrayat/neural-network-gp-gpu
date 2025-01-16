#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xcsv.hpp>
#include <vector>
#include <string>
#include <fstream>
#include "gradient_descent.cuh"
#include "model.hpp"
#include "utils.hpp"

using namespace std;
using namespace xt;

// Define class constructor
Model::Model(const vector<int> &architecture, const int &input_size) : architecture(architecture), input_size(input_size)
{
    // Define constructor for Model: architecture on the left is the object's member, on the right
    // is the constructor argument
    initialize_weights();
}

// Write class functions

void Model::initialize_weights()
// Set weights biases with random normal distribution
{
    weights.push_back(xt::random::randn<float>({architecture[0], input_size}));
    biases.push_back(xt::random::randn<float>({architecture[0], 1}));

    for (size_t l = 1; l < architecture.size(); l++) {
        xt::xarray<float> w = xt::random::randn<float>({architecture[l], architecture[l - 1]});
        xt::xarray<float> b = xt::random::randn<float>({architecture[l], 1});
        weights.push_back(w);
        biases.push_back(b);
    }
}

void Model::load_weights(const string &path)
{
    for (size_t l = 0; l < weights.size(); ++l) {
        ifstream w_infile(path + "/" + "weights_" + to_string(l) + ".csv");
        ifstream b_infile(path + "/" + "biases_" + to_string(l) + ".csv");

        weights[l] = xt::load_csv<float>(w_infile);
        biases[l] = xt::load_csv<float>(b_infile);

        w_infile.close();
        b_infile.close();
    }
}

void Model::load_loss(const string &pretrained_model_path)
{
    ifstream infile(pretrained_model_path + "/loss.csv");
    
    // Clear the current loss history to avoid appending
    loss_history.clear();
    
    // Read each line from the file and add to the loss_history vector
    string line;
    while (getline(infile, line)) {
        try {
            // Convert line to float and push to vector
            float loss = stod(line);
            loss_history.push_back(loss);
        } catch(const invalid_argument &e) {
            cerr << "Error: Invalid value in file '" << pretrained_model_path + "/loss.csv" << "' - " << line << endl;
            continue;
        }
    }
    infile.close();
}

void Model::save_weights(const string &path) const
{
    // Dump the weights and biases (for each layer)
    for (size_t l = 0; l < weights.size(); ++l) {
        string w_filepath = path + "/" + "weights_" + to_string(l) + ".csv";
        string b_filepath = path + "/" + "biases_" + to_string(l) + ".csv";
        ofstream w_outfile(path + "/" + "weights_" + to_string(l) + ".csv");
        ofstream b_outfile(path + "/" + "biases_" + to_string(l) + ".csv");
        check_iostream_state(w_outfile, w_filepath);
        check_iostream_state(b_outfile, b_filepath);

        xt::dump_csv(w_outfile, weights[l]);
        xt::dump_csv(b_outfile, biases[l]);

        w_outfile.close();
        b_outfile.close();
    }
}

void Model::save_loss(const string &path) const
{
    ofstream outfile(path + "/" + "loss.csv");
    // Write each element of the vector to the file
    for (const float &loss : loss_history) {
        outfile << loss << "\n";
    }
    outfile.close();
}

xarray<float> Model::predict(const xarray<float> &x_test) const
{
    xarray<float> a = xt::transpose(x_test);
    for (size_t l = 0; l < weights.size(); ++l)
    {
        a = sigmoid(xt::linalg::dot(weights[l], a) + biases[l]);
    }
    return xt::transpose(a);
}
