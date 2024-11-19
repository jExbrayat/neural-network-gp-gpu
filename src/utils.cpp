#include <iostream>
#include <xtensor/xarray.hpp>
#include <nlohmann/json.hpp>
using namespace std;
using namespace xt;

xarray<double> sigma(xarray<double> x)
{
    return 1 / (1 + xt::exp(-x));
}

xarray<double> sigma_derivative(xarray<double> x)
{
    return sigma(x) * (1 - sigma(x));
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