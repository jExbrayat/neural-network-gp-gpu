#include <iostream>
#include <xtensor/xarray.hpp>
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