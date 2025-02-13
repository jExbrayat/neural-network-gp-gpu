#include <cuda.h>

#define TILE_SIZE 16  // Does it must match the block dimension ? 

__global__ void matrixMulKernel(const float* A, const float* B, float* C, 
                                const int A_rows, const int A_cols, const int B_cols) {
    __shared__ float Asub[TILE_SIZE][TILE_SIZE];
    __shared__ float Bsub[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    float value = 0.0f;

    for (int t = 0; t < (A_cols + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load A and B tiles into shared memory
        if (row < A_rows && (t * TILE_SIZE + tx) < A_cols) {
            Asub[ty][tx] = A[row * A_cols + (t * TILE_SIZE + tx)];
        } else {
            Asub[ty][tx] = 0.0f;
        }

        if ((t * TILE_SIZE + ty) < A_cols && col < B_cols) {
            Bsub[ty][tx] = B[(t * TILE_SIZE + ty) * B_cols + col];
        } else {
            Bsub[ty][tx] = 0.0f;
        }

        __syncthreads();  // Ensure all threads have loaded data to avoid bank conflicts

        // Compute partial sum
        for (int k = 0; k < TILE_SIZE; ++k) {
            value += Asub[ty][k] * Bsub[k][tx];
        }

        __syncthreads();  // Ensure all threads have completed computation
    }

    if (row < A_rows && col < B_cols) {
        C[row * B_cols + col] = value;
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

/**
 * @brief 
 * 
 * @param input 
 * @param output 
 * @param rows Row number of the input matrix 
 * @param cols Column number of the input matrix
 * @return __global__ 
 */
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

__global__ void matrixPowerTwo(const float *matrix, float * output, const int rows, const int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows) {
        output[idy * cols + idx] = matrix[idy * cols + idx] * matrix[idy * cols + idx];        
    }    
}