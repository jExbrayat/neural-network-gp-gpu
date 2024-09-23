#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp> 
#include <xtensor/xview.hpp>
#include <xtensor/xrandom.hpp>
using namespace xt::placeholders;  // to enable _ syntax
using namespace std;
using namespace xt;
#include "utils.cpp"


int main()
{

    // Create two random datasets with different caracteristics
    auto x1 = create_random_dataset(0, 1.4, 500);
    xt::xarray<int> y1 = xt::ones<int>({500, 1});
    xt::xarray<double> dataset1 = xt::concatenate(xt::xtuple(x1, y1), 1);

    auto x2 = create_random_dataset(5, 0.8, 500);
    xt::xarray<int> y2 = xt::zeros<int>({500, 1});
    xt::xarray<double> dataset2 = xt::concatenate(xt::xtuple(x2, y2), 1);

    // Concatenate the two datasets and shuffle them
    xt::xarray<double> dataset = xt::concatenate(xtuple(dataset1, dataset2), 0);

    // Shuffle
    shuffleArray(dataset); 

    // Split into train and test sets
    xt::xarray<double> x_train = xt::view(dataset, xt::range(_, 800), xt::range(0, 2));
    xt::xarray<double> y_train = xt::view(dataset, xt::range(_, 800), 2);
    
    xt::xarray<double> x_test = xt::view(dataset, xt::range(800, _), xt::range(0, 2));
    xt::xarray<double> y_test = xt::view(dataset, xt::range(800, _), 2);

    // Display a sample of the dataset in the console
    for (int i=0; i < 10; i++) {
        std::cout << xt::view(x_train, i, xt::range(0, 2))
        << xt::view(y_train, i)
        << std::endl;
    }

    gnu_plot(x_train);


    return 0;
}