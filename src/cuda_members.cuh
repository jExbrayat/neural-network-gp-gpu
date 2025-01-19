#ifndef CUDA_MEMBERS_CUH
#define CUDA_MEMBERS_CUH

#include "cuda_utils.cuh"
#include <xtensor/xarray.hpp>
#include <iostream>
using namespace std;
using namespace xt;

struct CudaMemberVectors {
    // Class members for cuda
    vector<CudaMatrixMemory> weights; 
    vector<CudaMatrixMemory> biases;
    vector<CudaMatrixMemory> layer_outputs;  
    vector<CudaMatrixMemory> layer_activations; 

    vector<CudaMatrixMemory> deltas;
    vector<CudaMatrixMemory> grad_weights;
    vector<CudaMatrixMemory> grad_biases;
};

class Xtensor2CudaMatrixMemory {
    public:
    Xtensor2CudaMatrixMemory(const xarray<float>&x_train, const xarray<float>&y_train,  vector<xarray<float>>& weights, vector<xarray<float>>& biases,const  int batch_size);
    CudaMemberVectors CudaMembers;
    CudaMatrixMemory x;
    CudaMatrixMemory y;
};

#endif