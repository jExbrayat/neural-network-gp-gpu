#ifndef UTILS_HPP
#define UTILS_HPP
#include <xtensor/xarray.hpp>
#include <iostream>
#include <xtensor/xio.hpp>
using namespace xt;
using namespace std;

xarray<double> sigma(xarray<double> x);
xarray<double> sigma_derivative(xarray<double> x);
void shuffleArray(xt::xarray<double>& array);
xt::xarray<double> create_random_dataset(float mean, float variance, int n_observations);
void gnu_plot(xarray<double> two_dimensional_dataset);
std::tuple<std::vector<xarray<double>>, std::vector<xarray<double>>, xarray<double>>
make_gradient_descent(
    xarray<double> x_train,
    xarray<double> y_train, // shape must be (n, 1)
    int epochs,
    float learning_rate,
    std::vector<int> neurons_per_layer); // list of neurons in each layer
xarray<uint8_t> transform_mnist_images(vector<vector<uint8_t>> x, std::array<size_t, 2> shape);
xarray<int> transform_mnist_labels(vector<uint8_t> y, array<size_t, 2> shape);

#endif // UTILS_HPP