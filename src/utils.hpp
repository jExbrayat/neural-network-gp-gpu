#ifndef UTILS_HPP
#define UTILS_HPP

#include <iostream>
#include <xtensor/xarray.hpp>
#include <nlohmann/json.hpp>
#include <xtensor/xadapt.hpp>
using namespace std;
using namespace xt;

xarray<float> sigmoid(xarray<float> x);
xarray<float> sigmoid_derivative(xarray<float> x);
void print_shapes(xarray<float> &array, string msg);
void print_carray(float *carray, int rows, int cols, string msg);
nlohmann::json read_json(const string &config_file_path);
void scale_data(xarray<float> &x);
xarray<uint8_t> transform_mnist_images(vector<vector<uint8_t>> &x, std::array<size_t, 2> shape);
xarray<int> transform_mnist_labels(vector<uint8_t> &y, array<size_t, 2> shape);
void check_iostream_state(std::ios& iofile, const std::string& iofilepath);
class ArrayHandler
{
public:
    // Constructor
    ArrayHandler();
    // Destructor
    ~ArrayHandler();
    
    // Class members
    xarray<float> xtarray;
    float *carray;
    int rows;
    int cols; 

    // Methods
    void cast_xtarray(const xarray<float> &xtarray);
    void cast_carray(const float* carray, const int rows, const int cols);
};

#endif // UTILS_HPP