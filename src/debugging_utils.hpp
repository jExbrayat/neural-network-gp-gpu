#ifndef DEBUGGING_UTILS_HPP
#define DEBUGGING_UTILS_HPP

#include "cuda_utils.cuh"

void printCudaMatrixShapes(const CudaMatrixMemory &cudaMatrixMemory, string msg);

#endif