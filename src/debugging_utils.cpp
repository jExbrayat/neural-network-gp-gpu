#include <iostream>
#include "cuda_utils.cuh"

using namespace std;

void printCudaMatrixShapes(const CudaMatrixMemory &cudaMatrixMemory, string msg) {
    cout << msg << endl;
    cout << cudaMatrixMemory.rows << ", " << cudaMatrixMemory.cols << endl; 
}
