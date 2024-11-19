#ifndef UTILS_HPP
#define UTILS_HPP

#include <iostream>
#include <xtensor/xarray.hpp>
using namespace std;
using namespace xt;

xarray<double> sigma(xarray<double> x);
xarray<double> sigma_derivative(xarray<double> x);
void print_shapes(xarray<double> &array, string msg);


#endif // UTILS_HPP