#ifndef UTILS_HPP
#define UTILS_HPP

#include <iostream>
#include <xtensor/xarray.hpp>
#include <nlohmann/json.hpp>
using namespace std;
using namespace xt;

xarray<double> sigma(xarray<double> x);
xarray<double> sigma_derivative(xarray<double> x);
void print_shapes(xarray<double> &array, string msg);
nlohmann::json read_json(const string &config_file_path);


#endif // UTILS_HPP