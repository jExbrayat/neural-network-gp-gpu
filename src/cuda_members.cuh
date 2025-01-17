#ifndef CUDA_MEMBERS_CUH
#define CUDA_MEMBERS_CUH

#include "cuda_utils.cuh"
#include <iostream>
using namespace std;

struct CudaMemberVectors {
    // Class members for cuda
    vector<CudaMatrixMemory> weights; 
    vector<CudaMatrixMemory> biases;
    vector<CudaMatrixMemory> layer_outputs;  
    vector<CudaMatrixMemory> layer_activations; 

    vector<CudaMatrixMemory> deltas;
};

#endif