#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xmath.hpp>
#include <nlohmann/json.hpp>
#include <xtensor/xadapt.hpp>
using namespace std;
using namespace xt;

xarray<double> sigmoid(xarray<double> x)
{
    return 1 / (1 + xt::exp(-x));
}

xarray<double> sigmoid_derivative(xarray<double> x)
{
    return sigmoid(x) * (1 - sigmoid(x));
}

void print_shapes(xarray<double> &array, string msg) {
    cout << msg << endl;
    cout << array.shape(0) << ", " << array.shape(1) << endl;
}

nlohmann::json read_json(const string &config_file_path) {
    std::ifstream file(config_file_path);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open config.json file.\n";
        return 1;
    }
    nlohmann::json config;
    file >> config;
    file.close();
    return config;
}

xarray<int> transform_mnist_labels(vector<uint8_t> &y, array<size_t, 2> shape) {
    
    xt::xarray<uint8_t> y_tensor = xt::adapt(y, shape);
    return y_tensor;
}

xarray<uint8_t> transform_mnist_images(vector<vector<uint8_t>> &x, std::array<size_t, 2> shape) {
    
    // Flatten the 2D vector into 1D vector
    std::vector<uint8_t> flat_data;
    for (const auto& image : x) {
        flat_data.insert(flat_data.end(), image.begin(), image.end());
    }

    // Create an xtensor with the flattened data and reshape it
    xt::xarray<uint8_t> x_tensor = xt::adapt(flat_data, shape);

    return x_tensor;
}

void scale_data(xarray<double> &x) {
    // Scale data in [0; 1]
    x = (x - xt::amin(x)()) / (xt::amax(x)() - xt::amin(x)());
}

void check_iostream_state(std::ios& iofile, const std::string& iofilepath) {
    if (!iofile) {  // If the stream is in a bad state (i.e., failed to open)
        throw std::runtime_error("Failed to open file: " + iofilepath);
    }
}
