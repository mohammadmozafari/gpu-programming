#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

// GPU kernel
__global__ void add(float* a, float* b, float* c, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        c[index] = a[index] + b[index];
    }
}

// CPU implementation
void add_cpu(float* a, float* b, float* c, int N) {
    for (int i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int N = 1 << 24;                        // ~16 million elements
    size_t size = N * sizeof(float);

    float *a = (float*)malloc(size);
    float *b = (float*)malloc(size);
    float *c_gpu = (float*)malloc(size);
    float *c_cpu = (float*)malloc(size);
    float gpu_time = 0.0f;

    // Seed RNG
    srand((unsigned int)time(NULL));

    // Random initialization
    for (int i = 0; i < N; i++) {
        a[i] = (float)(rand() % 1000) / 100.0f;  // values in [0,10)
        b[i] = (float)(rand() % 1000) / 100.0f;
    }

    // --- CPU timing
    clock_t start_cpu = clock();
    add_cpu(a, b, c_cpu, N);
    clock_t end_cpu = clock();
    double cpu_time = (double)(end_cpu - start_cpu) / CLOCKS_PER_SEC;

    // --- GPU memory allocation
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // --- GPU timing (CUDA events)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
    cudaEventRecord(start);
    add<<<blocks_per_grid, threads_per_block>>>(d_a, d_b, d_c, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    
    cudaMemcpy(c_gpu, d_c, size, cudaMemcpyDeviceToHost);
    
    // --- Check correctness
    for (int i = 0; i < N; i++) {
        if (fabs(c_cpu[i] - c_gpu[i]) > 1e-6) {
            printf("Mismatch at index %d!\n", i);
        }
    }

    // --- Print timing
    printf("\nCPU time: %f ms\n", cpu_time * 1000);
    printf("GPU time: %f ms\n", gpu_time);

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c_cpu);
    free(c_gpu);

    return 0;
}
