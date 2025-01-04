#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

using namespace std;

// CUDA kernels
__global__ void sigmoidKernel(double* input, double* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = 1.0 / (1.0 + exp(-input[idx]));
    }
}

__global__ void sigmoidDerivativeKernel(double* input, double* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        double sigmoid_val = 1.0 / (1.0 + exp(-input[idx]));
        output[idx] = sigmoid_val * (1.0 - sigmoid_val);
    }
}

void gpuMatrixMultiply(cublasHandle_t handle, double* d_A, double* d_B, double* d_C, int m, int n, int k) {
    const double alpha = 1.0;
    const double beta = 0.0;
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_B, n, d_A, k, &beta, d_C, n);
}

class GradientDescent {
private:
    double* d_x_train;
    double* d_y_train;
    vector<double*> weights; 
    vector<double*> biases;  
    vector<double*> layer_outputs;
    vector<double*> layer_activations;

    int num_layers;
    int dataset_size;
    cublasHandle_t handle;

public:
    GradientDescent(const double* h_x_train, const double* h_y_train, int dataset_size,
                    vector<vector<double>>& h_weights, vector<vector<double>>& h_biases)
        : dataset_size(dataset_size) {
        cublasCreate(&handle);

        cudaMalloc(&d_x_train, dataset_size * sizeof(double));
        cudaMalloc(&d_y_train, dataset_size * sizeof(double));
        cudaMemcpy(d_x_train, h_x_train, dataset_size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y_train, h_y_train, dataset_size * sizeof(double), cudaMemcpyHostToDevice);

        num_layers = h_weights.size();
        for (int l = 0; l < num_layers; ++l) {
            double* d_weight;
            double* d_bias;

            cudaMalloc(&d_weight, h_weights[l].size() * sizeof(double));
            cudaMalloc(&d_bias, h_biases[l].size() * sizeof(double));

            cudaMemcpy(d_weight, h_weights[l].data(), h_weights[l].size() * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(d_bias, h_biases[l].data(), h_biases[l].size() * sizeof(double), cudaMemcpyHostToDevice);

            weights.push_back(d_weight);
            biases.push_back(d_bias);
        }
    }

    void forward_pass(const double* d_x_batch) {
        layer_activations.resize(num_layers + 1);
        layer_outputs.resize(num_layers);

        layer_activations[0] = const_cast<double*>(d_x_batch);

        for (int l = 0; l < num_layers; ++l) {
            int m = 1, n = 1, k = 1; 
            cudaMalloc(&layer_outputs[l], m * n * sizeof(double));
            gpuMatrixMultiply(handle, weights[l], layer_activations[l], layer_outputs[l], m, n, k);

            dim3 blockSize(256);
            dim3 gridSize((m * n + blockSize.x - 1) / blockSize.x);
            cudaMalloc(&layer_activations[l + 1], m * n * sizeof(double));
            sigmoidKernel<<<gridSize, blockSize>>>(layer_outputs[l], layer_activations[l + 1], m * n);
        }
    }

    void backward_pass(const double* d_y_batch, const int& current_batch_size, const float& learning_rate) {
        vector<double*> deltas(num_layers);

        int size = current_batch_size;

        cudaMalloc(&deltas[num_layers - 1], size * sizeof(double));
        sigmoidDerivativeKernel<<<(size + 255) / 256, 256>>>(layer_outputs[num_layers - 1], deltas[num_layers - 1], size);

        for (int l = num_layers - 2; l >= 0; --l) {
            cudaMalloc(&deltas[l], size * sizeof(double));
            sigmoidDerivativeKernel<<<(size + 255) / 256, 256>>>(layer_outputs[l], deltas[l], size);
        }

        for (int l = 0; l < num_layers; ++l) {
            gpuMatrixMultiply(handle, deltas[l], layer_activations[l], weights[l], size, size, size);
        }
    }

    void train(const unsigned int& epochs, const int& batch_size, const float& learning_rate) {
        int batch_number = dataset_size / batch_size;

        for (unsigned int epoch = 0; epoch < epochs; ++epoch) {
            cout << "Epoch: " << epoch << endl;

            for (int batch_start = 0; batch_start < dataset_size; batch_start += batch_size) {
                int current_batch_size = min(batch_size, dataset_size - batch_start);

                const double* d_x_batch = d_x_train + batch_start;
                const double* d_y_batch = d_y_train + batch_start;

                forward_pass(d_x_batch);
                backward_pass(d_y_batch, current_batch_size, learning_rate);
            }
        }
    }

    ~GradientDescent() {
        cudaFree(d_x_train);
        cudaFree(d_y_train);
        for (int l = 0; l < num_layers; ++l) {
            cudaFree(weights[l]);
            cudaFree(biases[l]);
        }
        cublasDestroy(handle);
    }
};
