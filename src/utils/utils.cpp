#include <iostream>
#include <random>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xadapt.hpp>
#include <fstream>
#include <sstream>
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
    gnuplot_commands << "set terminal wxt size 1200,800\n";                           // Set window size
    gnuplot_commands << "set xlabel 'x1'\n";                                          // Set x-axis label
    gnuplot_commands << "set ylabel 'x2'\n";                                          // Set y-axis label
    gnuplot_commands << "set title '" + plot_title + "'\n";                           // Set plot title
    gnuplot_commands << "set pointsize 1.5\n";                                        // Set point size
    gnuplot_commands << "set palette defined (0 'blue', 1 'red')\n";                  // Define color palette for clusters
    gnuplot_commands << "plot 'temp/gnuplot.dat' u 1:2:3 with points palette pt 7\n"; // Plot data points using the third column for color

    // Close the file
    gnuplot_commands.close();

    // Run gnuplot with the command file
    std::system(("unset GTK_PATH && gnuplot-x11 -persist " + gnuplot_commands_path).c_str());
}

void gnuplot_ypred_ytrue(const xt::xarray<double> &y_true, const xt::xarray<double> &y_pred, const std::string &plot_title)
{
    // Define paths
    std::string gnuplot_commands_path = "temp/gnuplot_commands.gp";
    std::string data_path = "temp/gnuplot.dat";

    // Retrieve dataset size (assuming y_true and y_pred are both (n, 1) shape)
    int n_rows = y_true.shape()[0];

    // Create a data file
    std::ofstream data(data_path);
    for (int i = 0; i < n_rows; i++)
    {
        data << y_true(i, 0) << " " << y_pred(i, 0) << "\n";
    }
    data.close();

    // Create Gnuplot command file
    std::ofstream gnuplot_commands(gnuplot_commands_path);
    gnuplot_commands << "set terminal wxt size 1200,800\n";                             // Set window size
    gnuplot_commands << "set xlabel 'y_true'\n";                                        // Set x-axis label
    gnuplot_commands << "set ylabel 'y_pred'\n";                                        // Set y-axis label
    gnuplot_commands << "set title '" + plot_title + "'\n";                             // Set plot title
    gnuplot_commands << "set pointsize 1.5\n";                                          // Set point size
    gnuplot_commands << "plot 'temp/gnuplot.dat' using 1:2 with points pt 7 notitle\n"; // Plot y_true vs y_pred

    // Close the Gnuplot command file
    gnuplot_commands.close();

    // Execute Gnuplot
    std::system(("gnuplot -persist " + gnuplot_commands_path).c_str());
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
    gnuplot_commands << "set terminal wxt size 1200,800\n";                                         // Set window size
    gnuplot_commands << "set xlabel 'Iteration'\n";                                                 // Set x-axis label
    gnuplot_commands << "set ylabel 'Loss'\n";                                                      // Set y-axis label
    gnuplot_commands << "set title '" + plot_title + "'\n";                                         // Set plot title
    gnuplot_commands << "set grid\n";                                                               // Enable grid
    gnuplot_commands << "set style line 1 lc rgb '#0060ad' lw 2\n";                                 // Define line style
    gnuplot_commands << "plot '" + data_path + "' using 1:2 with lines linestyle 1 title 'Loss'\n"; // Plot the data as a line

    gnuplot_commands.close();

    // Execute the Gnuplot command file
    std::system(("gnuplot -persist " + gnuplot_commands_path).c_str());
}

void gnuplot_image_plot(xt::xarray<double> image_values, std::string plot_title)
{
    // Define paths for Gnuplot commands and data
    std::string gnuplot_commands_path = "temp/gnuplot_image_commands.gp";
    std::string data_path = "temp/gnuplot_image.dat";

    // Retrieve dataset size
    int img_size = image_values.shape()[0];
    int img_dim = std::sqrt(img_size);  // Assume it's a square image

    // Create a data file for Gnuplot
    std::ofstream data(data_path);
    for (int i = 0; i < img_dim; ++i)
    {
        for (int j = 0; j < img_dim; ++j)
        {
            // Write the image values: coordinates (i, j), pixel intensity
            data << i << " " << j << " " << image_values(i * img_dim + j) << std::endl;
        }
        data << std::endl; // Separate rows with a blank line for Gnuplot
    }
    data.close();

    // Create the Gnuplot command file
    std::ofstream gnuplot_commands(gnuplot_commands_path);
    gnuplot_commands << "set terminal wxt size 1200,1200\n";                                       // Set window size
    gnuplot_commands << "set xlabel 'X-axis'\n";                                                   // Set x-axis label
    gnuplot_commands << "set ylabel 'Y-axis'\n";                                                   // Set y-axis label
    gnuplot_commands << "set title '" + plot_title + "'\n";                                        // Set plot title
    gnuplot_commands << "unset key\n";                                                             // Disable legend
    gnuplot_commands << "set size square\n";                                                       // Set plot to square
    gnuplot_commands << "set palette gray\n";                                                      // Set grayscale palette
    gnuplot_commands << "plot '" + data_path + "' using 1:2:3 with image\n";                       // Plot the image

    gnuplot_commands.close();

    // Execute the Gnuplot command file
    std::system(("gnuplot -persist " + gnuplot_commands_path).c_str());
}


void shuffleArray(xt::xarray<double> &array)
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


xarray<uint8_t> transform_mnist_images(vector<vector<uint8_t>> x, std::array<size_t, 2> shape) {
    
    // Flatten the 2D vector into 1D vector
    std::vector<uint8_t> flat_data;
    for (const auto& image : x) {
        flat_data.insert(flat_data.end(), image.begin(), image.end());
    }

    // Create an xtensor with the flattened data and reshape it
    xt::xarray<uint8_t> x_tensor = xt::adapt(flat_data, shape);

    return x_tensor;
}

xarray<int> transform_mnist_labels(vector<uint8_t> y, array<size_t, 2> shape) {
    
    xt::xarray<uint8_t> y_tensor = xt::adapt(y, shape);
    return y_tensor;
}

xarray<double> scale_data(xarray<double> x) {
    return (x - xt::amin(x)()) / (xt::amax(x)() - xt::amin(x)());
}

// Function to save an xtensor xarray to a CSV file
void save_matrix_to_csv(const xt::xarray<double>& array, const std::string& file_path) {
    std::ofstream file(file_path);
    if (file.is_open()) {
        for (size_t i = 0; i < array.shape(0); ++i) {
            for (size_t j = 0; j < array.shape(1); ++j) {
                file << array(i, j);
                if (j != array.shape(1) - 1) {
                    file << ",";  // Add comma between elements
                }
            }
            file << "\n";  // New line at the end of each row
        }
        file.close();
    } else {
        std::cerr << "Error opening file " << file_path << std::endl;
    }
}

void save_unidim_array_to_csv(const xt::xarray<double>& array, const std::string& file_path) {
    std::ofstream file(file_path);
    if (file.is_open()) {
        for (size_t i = 0; i < array.shape(0); ++i) {
            file << array(i);
            file << ",";  // Add comma between elements
        }
        file.close();
    } else {
        std::cerr << "Error opening file " << file_path << std::endl;
    }
}

// Function to dump the vectors of weights and biases (for each layer)
void dump_model(const std::vector<xt::xarray<double>>& weights, const std::vector<xt::xarray<double>>& biases, const xt::xarray<double>& mse_array, string dir_path) {
    for (size_t i = 0; i < weights.size(); ++i) {
        save_matrix_to_csv(weights[i], dir_path + "/" + "weights_layer_" + std::to_string(i) + ".csv");
        save_matrix_to_csv(biases[i], dir_path + "/" + "biases_layer_" + std::to_string(i) + ".csv");
    }
    save_unidim_array_to_csv(mse_array, dir_path + "/" + "loss.csv");
}

// Function to load an xtensor xarray from a CSV file
// The result is a two dimensional array, i.e. a matrix
xt::xarray<double> load_xarray_from_csv(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Error opening file " << file_path << std::endl;
        return xt::xarray<double>();
    }

    std::vector<std::vector<double>> data;
    std::string line;
    while (std::getline(file, line)) {
        std::vector<double> row;
        std::stringstream line_stream(line);
        std::string cell;
        while (std::getline(line_stream, cell, ',')) {
            row.push_back(std::stod(cell));
        }
        data.push_back(row);
    }
    file.close();

    // Convert vector of vectors to xtensor xarray
    size_t rows = data.size();
    size_t cols = data[0].size();
    xt::xarray<double> array = xt::zeros<double>({rows, cols});
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            array(i, j) = data[i][j];
        }
    }
    return array;
}

// Function to load the model weights, biases, and mse array
void load_model(std::vector<xt::xarray<double>>& weights, std::vector<xt::xarray<double>>& biases, xt::xarray<double>& mse_array, const std::string& dir_path, const int num_layers) {
    size_t i = 0;
    while (true) {
        std::string weight_file = dir_path + "/" + "weights_layer_" + std::to_string(i) + ".csv";
        std::string bias_file = dir_path + "/" + "biases_layer_" + std::to_string(i) + ".csv";

        // Check if the weight and bias files exist
        std::ifstream wfile(weight_file);
        std::ifstream bfile(bias_file);
        if (!wfile.is_open() || !bfile.is_open()) break;  // Stop when no more layers are found
        
        // Check if the number of layers in the model matches the expected architecture
        if (i >= num_layers) {
            std::cerr << "Error when loading pre-trained model. Check that the architecture specified in config.json "
                      << "corresponds to the architecture of the selected pre-trained model." << std::endl;
            std::exit(EXIT_FAILURE); // Exit the application with a failure status
        }
        
        // Load weight and bias arrays
        weights.push_back(load_xarray_from_csv(weight_file));
        biases.push_back(load_xarray_from_csv(bias_file));
        
        wfile.close();
        bfile.close();
        ++i;
    }

    // Load the loss array (mse_array)
    mse_array = load_xarray_from_csv(dir_path + "/" + "loss.csv");
    mse_array.reshape({mse_array.size()}); // Force the array to be unidimensional
}

