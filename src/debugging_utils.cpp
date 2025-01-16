#include <iostream>
#include <xtensor/xarray.hpp>
#include <cmath>
#include "cuda_utils.cuh"
#include "utils.hpp"

using namespace std;

void printCudaMatrixShapes(const CudaMatrixMemory &cudaMatrixMemory, string msg) {
    cout << msg << endl;
    cout << cudaMatrixMemory.rows << ", " << cudaMatrixMemory.cols << endl; 
}
void checkCudaComputation(CudaMatrixMemory &cuda_array, xt::xarray<float> &reference_xtarray, float epsilon, string custom_msg) {
    // Check if the shape of reference matches the shape of cuda_array
    if (reference_xtarray.shape(0) != cuda_array.rows || reference_xtarray.shape(1) != cuda_array.cols) {
        std::cout << "The shapes of reference and target do not match:" << std::endl;
        print_shapes(reference_xtarray, "   Reference: ");
        std::cout << "   Target: " << cuda_array.rows << ", " << cuda_array.cols << std::endl;
    }
    

    // Copy the device array back to host
    float *carray = cuda_array.allocAndSend2Host(); // now carray (host array) contains the device values

    // Compare element wise
    int error_counter = 0; // Init counter of errors i.e. # values different from the reference +- epsilon
    for (size_t i = 0; i < cuda_array.rows; i++) {
        for (size_t j = 0; j < cuda_array.cols; j++) {
            float abs_difference = abs(reference_xtarray(i, j) - carray[i * cuda_array.cols + j]);
            if (abs_difference > epsilon) {
                error_counter += 1;
            }
        }
    }

    float error_rate = error_counter / (cuda_array.rows * cuda_array.cols);

    // Print the error count
    cout << custom_msg << endl;
    cout << "The number of error +- epsilon for this computation is: " << error_counter << endl;
    cout << "The rate of error +- epsilon for this computation is: " << error_rate << endl; 
}
