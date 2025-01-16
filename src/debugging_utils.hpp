#ifndef DEBUGGING_UTILS_HPP
#define DEBUGGING_UTILS_HPP

#include "cuda_utils.cuh"
#include <xtensor/xarray.hpp>

void printCudaMatrixShapes(const CudaMatrixMemory &cudaMatrixMemory, string msg);
void checkCudaComputation(CudaMatrixMemory &cuda_array, xt::xarray<float> &reference_xtarray, float epsilon, string custom_msg);

#endif