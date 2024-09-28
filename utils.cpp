#include <iostream>
#include <random>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xrandom.hpp>
#include <fstream>
using namespace xt::placeholders; // to enable _ syntax
using namespace std;
using namespace xt;

void gnuplot(xarray<double> two_dimensional_dataset, xarray<int> true_pred, string plot_title) // true_pred shape: (n, 1)
{

    // Define paths
    string gnuplot_commands_path = "temp/gnuplot_commands.gp";
    string data_path = "temp/gnuplot.dat";

    // Retrieve dataset size
    int n_rows = two_dimensional_dataset.shape()[0];

    // Create a data file
    std::ofstream data(data_path);
    for (int i = 0; i < n_rows; i++)
    {
        data
            << two_dimensional_dataset(i, 0) << " "
            << two_dimensional_dataset(i, 1) << " "
            << true_pred(i, 0)
            << endl; // x and y values
    }
    data.close();

    // Use Gnuplot to plot the data
    std::ofstream gnuplot_commands(gnuplot_commands_path);
    gnuplot_commands << "set terminal wxt size 1200,800\n";         // Set window size
    gnuplot_commands << "set xlabel 'x1'\n";                        // Set x-axis label
    gnuplot_commands << "set ylabel 'x2'\n";                        // Set y-axis label
    gnuplot_commands << "set title '" + plot_title + "'\n"; // Set plot title
    gnuplot_commands << "set pointsize 1.5\n";                        // Set point size
    gnuplot_commands << "set palette defined (0 'blue', 1 'red')\n"; // Define color palette for clusters
    gnuplot_commands << "plot 'temp/gnuplot.dat' u 1:2:3 with points palette pt 7\n"; // Plot data points using the third column for color

    // Close the file
    gnuplot_commands.close();

    // Run gnuplot with the command file
    std::system(("unset GTK_PATH && gnuplot-x11 -persist " + gnuplot_commands_path).c_str());
}

void gnuplot_loss_plot(xt::xarray<double> loss_values, std::string plot_title)
{
    // Define paths for Gnuplot commands and data
    std::string gnuplot_commands_path = "temp/gnuplot_loss_commands.gp";
    std::string data_path = "temp/gnuplot_loss.dat";

    // Retrieve dataset size
    int n_points = loss_values.shape()[0];

    // Create a data file for Gnuplot
    std::ofstream data(data_path);
    for (int i = 0; i < n_points; ++i)
    {
        // Write each point: iteration (i), loss_value
        data << i << " " << loss_values(i) << std::endl;
    }
    data.close();

    // Create the Gnuplot command file
    std::ofstream gnuplot_commands(gnuplot_commands_path);
    gnuplot_commands << "set terminal wxt size 1200,800\n";         // Set window size
    gnuplot_commands << "set xlabel 'Iteration'\n";                 // Set x-axis label
    gnuplot_commands << "set ylabel 'Loss'\n";                      // Set y-axis label
    gnuplot_commands << "set title '" + plot_title + "'\n";         // Set plot title
    gnuplot_commands << "set grid\n";                               // Enable grid
    gnuplot_commands << "set style line 1 lc rgb '#0060ad' lw 2\n"; // Define line style
    gnuplot_commands << "plot '" + data_path + "' using 1:2 with lines linestyle 1 title 'Loss'\n"; // Plot the data as a line

    gnuplot_commands.close();

    // Execute the Gnuplot command file
    std::system(("gnuplot -persist " + gnuplot_commands_path).c_str());
}

    xt::xarray<double> create_random_dataset(float mean, float variance, int n_observations)
    {
        // Define seed
        std::random_device rd;
        std::mt19937 e2(rd());

        std::normal_distribution<> dist(mean, variance);

        // Init data array
        xt::xarray<double> dataset = xt::zeros<double>({0, 2});
        for (int n = 0; n < n_observations; n++)
        {
            xt::xarray<double> data_point = {dist(e2), dist(e2)};
            data_point.reshape({1, 2});
            dataset = xt::concatenate(xtuple(dataset, data_point), 0);
        }

        return dataset;
    }

void shuffleArray(xt::xarray<double> & array)
{
    // Get the number of rows and columns
    std::size_t rows = array.shape(0);
    std::size_t cols = array.shape(1);

    // Create an index array to shuffle
    xt::xarray<std::size_t> indices = xt::arange<std::size_t>(rows);

    // Shuffle the indices
    xt::random::shuffle(indices);

    // Create a temporary array to hold the shuffled rows
    xt::xarray<double> temp = xt::empty_like(array);

    // Fill the temporary array with shuffled rows
    for (std::size_t i = 0; i < rows; ++i)
    {
        for (std::size_t j = 0; j < cols; ++j)
        {
            temp(i, j) = array(indices(i), j);
        }
    }

    // Copy the shuffled rows back to the original array
    array = temp;
}

    xarray<double> sigma(xarray<double> x)
    {
        return 1 / (1 + xt::exp(-x));
    }

    xarray<double> sigma_derivative(xarray<double> x)
    {
        return sigma(x) * (1 - sigma(x));
    }
