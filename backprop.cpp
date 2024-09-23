#include <iostream>
#include <random>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp> 
#include <xtensor/xview.hpp>
#include <xtensor/xrandom.hpp>
#include <fstream>
using namespace xt::placeholders;  // to enable _ syntax
using namespace std;
using namespace xt;

void gnu_plot(xarray<double> two_dimensional_dataset) {
    
    // Retrieve dataset size
    int n_rows = two_dimensional_dataset.shape()[0];

    // Create a data file
    std::ofstream data("temp/gnu.dat");

    for (int i = 0; i < n_rows; i++) {
        data << two_dimensional_dataset(i, 0) << " " << two_dimensional_dataset(i, 1) << endl; // x and y values
    }
    data.close();

    // Use Gnuplot to plot the data
    system("gnuplot -p -e \"set terminal x11; plot 'temp/gnu.dat' using 1:2 with points\"");
}

xt::xarray<double> create_random_dataset(float mean, float variance, int n_observations) { 
    // Define seed
    std::random_device rd;
    std::mt19937 e2(rd());

    std::normal_distribution<> dist(mean, variance);

    // Init data array
    xt::xarray<double> dataset = xt::zeros<double>({0, 2}) ; 
    for (int n = 0; n < n_observations; n++)
    {
        xt::xarray<double> data_point = {dist(e2), dist(e2)};
        data_point.reshape({1, 2});
        dataset = xt::concatenate(xtuple(dataset, data_point), 0);
    }

    return dataset;
}

void shuffleArray(xt::xarray<double>& array) {
    // Get the number of rows
    std::size_t rows = array.shape(0);

    // Create an index array to shuffle
    xt::xarray<std::size_t> indices = xt::arange<std::size_t>(rows);

    // Shuffle the indices
    xt::random::shuffle(indices);

    // Create a temporary array to hold the shuffled rows
    xt::xarray<double> temp = xt::empty_like(array);

    // Fill the temporary array with shuffled rows
    for (std::size_t i = 0; i < rows; ++i) {
        temp(i, 0) = array(indices(i), 0);
        temp(i, 1) = array(indices(i), 1);
        temp(i, 2) = array(indices(i), 2);
    }

    // Copy the shuffled rows back to the original array
    array = temp;
}

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


    return 0;
}