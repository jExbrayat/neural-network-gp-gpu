#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xcsv.hpp>
#include <vector>
#include <string>
#include <fstream>
#include "src/include/gradient_descent.hpp"
#include "src/include/model.hpp"
#include "src/include/utils.hpp"

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
    weights.push_back(xt::random::randn<double>({architecture[0], input_size}));
    biases.push_back(xt::random::randn<double>({architecture[0], 1}));

    for (size_t l = 1; l < architecture.size(); l++) {
        xt::xarray<double> w = xt::random::randn<double>({architecture[l], architecture[l - 1]});
        xt::xarray<double> b = xt::random::randn<double>({architecture[l], 1});
        weights.push_back(w);
        biases.push_back(b);
    }
}

void Model::load_weights(const string &path)
{
    for (size_t l = 0; l < weights.size(); ++l) {
        ifstream w_infile(path + "/" + "weights_" + to_string(l) + ".csv");
        ifstream b_infile(path + "/" + "biases_" + to_string(l) + ".csv");

        weights[l] = xt::load_csv<double>(w_infile);
        biases[l] = xt::load_csv<double>(b_infile);

        w_infile.close();
        b_infile.close();
    }
}

void Model::load_loss(const string &loss_file_path)
{
    ifstream infile(loss_file_path);
    
    // Clear the current loss history to avoid appending
    loss_history.clear();
    
    // Read each line from the file and add to the loss_history vector
    string line;
    while (getline(infile, line)) {
        try {
            // Convert line to double and push to vector
            double loss = stod(line);
            loss_history.push_back(loss);
        } catch(const invalid_argument &e) {
            cerr << "Error: Invalid value in file '" << loss_file_path << "' - " << line << endl;
            continue;
        }
    }
    infile.close();
}

void Model::save_weights(const string &path) const
{
    // Dump the weights and biases (for each layer)
    for (size_t l = 0; l < weights.size(); ++l) {
        ofstream w_outfile(path + "/" + "weights_" + to_string(l) + ".csv");
        ofstream b_outfile(path + "/" + "biases_" + to_string(l) + ".csv");

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
    for (const double &loss : loss_history) {
        outfile << loss << "\n";
    }
    outfile.close();
}

xarray<double> Model::predict(const xarray<double> &x_test) const
{
    xarray<double> a = xt::transpose(x_test);
    for (size_t l = 0; l < weights.size(); ++l)
    {
        a = sigma(xt::linalg::dot(weights[l], a) + biases[l]);
    }
    return xt::transpose(a);
}
