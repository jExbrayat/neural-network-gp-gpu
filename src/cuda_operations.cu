#include <cuda.h>

__global__ void matrixMulKernel(const float* A, const float* B, float* C, const int A_rows, const int A_cols, const int B_cols) {
    // Compute the global thread index for both x and y dimensions
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Column index in C
    int idy = blockIdx.y * blockDim.y + threadIdx.y; // Row index in C

    // Check if the thread is within bounds
    if (idx < B_cols && idy < A_rows) {
        float value = 0.0f;
        // Perform the dot product for the row of A and column of B
        for (int k = 0; k < A_cols; ++k) {
            value += A[idy * A_cols + k] * B[k * B_cols + idx];
        }
        C[idy * B_cols + idx] = value;  // Store the result in C
    }
}

__global__ void addBiasToMatrixKernel(const float* matrix, const float* biases, float* result, int rows, int cols) {
    // Compute the global thread index for both x and y dimensions
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Column index
    int idy = blockIdx.y * blockDim.y + threadIdx.y; // Row index

    // Check if the thread is within bounds
    if (idx < cols && idy < rows) {
        // Add the bias to each element in the column
        result[idy * cols + idx] = matrix[idy * cols + idx] + biases[idy];
    }
}

/**
 * @brief Perform A + lambda B
 */
__global__ void addMatrixToMatrix(const float* A, const float* B, float lambda, float* result, int rows, int cols) {
    // Compute the global thread index for both x and y dimensions
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Column index
    int idy = blockIdx.y * blockDim.y + threadIdx.y; // Row index

    if (idx < cols && idy < rows) {
        result[idy * cols + idx] = A[idy * cols + idx] + lambda * B[idy * cols + idx];
    }
}

__global__ void sigmoidKernel(const float* input, float* output, const int rows, const int cols) {
    // Compute the global thread index for both x and y dimensions
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is within bounds
    if (idx < cols && idy < rows) {
        int index = idy * cols + idx;  // Convert 2D index to 1D
        output[index] = 1.0f / (1.0f + expf(-input[index]));  // Sigmoid function
    }
}

__device__ float _sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void sigmoidDerivativeKernel(const float* input, float* output, const int rows, const int cols) {
    
    // Compute the global thread index for both x and y dimensions
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is within bounds
    if (idx < cols && idy < rows) {
        int index = idy * cols + idx;  // Convert 2D index to 1D
        output[index] = _sigmoid(input[index]) * (1 - _sigmoid(input[index]));
    }
}


__global__ void transposeKernel(const float* input, float* output, const int rows, const int cols) {
    // Compute the global thread index for both x and y dimensions
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows) {
        int input_index = idy * cols + idx;
        int output_index = idx * rows + idy;
        output[output_index] = input[input_index];        
    }
}

__global__ void matMulElementWise(const float *A, const float *B, float *output, const int rows, const int cols) {
    
    // Compute the global thread index for both x and y dimensions
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows) {
        output[idy * cols + idx] = A[idy * cols + idx] * B[idy * cols + idx];        
    }
}

__global__ void matrixScalarKernel(const float *matrix, float* output, const float scalar, const int rows, const int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows) {
        output[idy * cols + idx] = matrix[idy * cols + idx] * scalar;        
    }
}

/**
 * @brief Compute mean across columns, i.e. resulting matrix has a unique column.
 * The kernel should be launched with a unidimensional grid (e,g, (0, 256)) since the grid navigates through columns only.
 */
__global__ void computeMeanKernel(const float *matrix, float * output, const int rows, const int cols) {
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idy < rows) {
        float sum = 0.f;
        for (int k = 0; k < cols; k++) {
            sum += matrix[idy * cols + k];
        }
        output[idy] = sum / cols;
    }
}
