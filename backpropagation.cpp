#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xrandom.hpp>
#include <tuple>
using namespace xt::placeholders; // to enable _ syntax
using namespace std;
using namespace xt;

std::tuple<
    xt::xarray<double>,
    xt::xarray<double>,
    xt::xarray<double>,
    xt::xarray<double>,
    xt::xarray<double>,
    xt::xarray<double>
>
make_gradient_descent(
    xarray<double> x_train,
    xarray<double> y_train,
    int epochs,
    float learning_rate
) {

    // Initialize network

    // Weights
    xarray<double> w1 = xt::random::randn<double>({3, 2});
    xarray<double> w2 = xt::random::randn<double>({5, 3});
    xarray<double> w3 = xt::random::randn<double>({1, 5});

    // Biasses
    xarray<double> b1 = xt::random::randn<double>({3, 1});
    xarray<double> b2 = xt::random::randn<double>({5, 1});
    xarray<double> b3 = xt::random::randn<double>({1, 1});

    // Define gradient matrices
    xarray<double> nabla_w1 = zeros<double>(w1.shape());
    xarray<double> nabla_w2 = zeros<double>(w2.shape());
    xarray<double> nabla_w3 = zeros<double>(w3.shape());

    return std::make_tuple(w1, w2, w3, b1, b2, b3);
}