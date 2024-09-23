#ifndef UTILS_HPP
#define UTILS_HPP
#include <xtensor/xarray.hpp>
using namespace xt;

xarray<double> sigma(xarray<double> x);
xarray<double> sigma_derivative(xarray<double> x);
void shuffleArray(xt::xarray<double>& array);
xt::xarray<double> create_random_dataset(float mean, float variance, int n_observations);
void gnu_plot(xarray<double> two_dimensional_dataset);

#endif // UTILS_HPP