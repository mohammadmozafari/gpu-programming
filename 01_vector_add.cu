#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>

#define N (1 << 20)       // ~1 million elements
#define THREADS_PER_BLOCK 256
#define CPU_RUNS 5
#define GPU_RUNS 10

// ---------------- CPU function ----------------
void vector_add_cpu(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// ---------------- GPU kernel ----------------
__global__ void vector_add_gpu(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// ---------------- Initialize arrays ----------------
void init_array(float* a, int n) {
    for (int i = 0; i < n; i++) {
        a[i] = (float)(rand() % 1000) / 100.0f;  // values in [0, 10)
    }
}

// ---------------- Benchmark CPU ----------------
double benchmark_cpu(const float* a, const float* b, float* c, int n, int runs) {
    double total_time = 0.0;
    for (int i = 0; i < runs; i++) {
        clock_t start = clock();
        vector_add_cpu(a, b, c, n);
        clock_t end = clock();
        total_time += (double)(end - start) / CLOCKS_PER_SEC;
    }
    return (total_time / runs) * 1000;  // return avg time in ms
}

// ---------------- Benchmark GPU ----------------
float benchmark_gpu(const float* d_a, const float* d_b, float* d_c, int n, int runs) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Warm-up
    vector_add_gpu<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    float total_time = 0.0f;
    for (int i = 0; i < runs; i++) {
        cudaEventRecord(start);
        vector_add_gpu<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float elapsed;
        cudaEventElapsedTime(&elapsed, start, stop);
        total_time += elapsed;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return total_time / runs;  // avg kernel execution time in ms
}

// ---------------- Check correctness ----------------
void check_result(const float* cpu, const float* gpu, int n) {
    for (int i = 0; i < n; i++) {
        if (fabs(cpu[i] - gpu[i]) > 1e-6) {
            printf("Mismatch at index %d! CPU=%f, GPU=%f\n", i, cpu[i], gpu[i]);
            return;
        }
    }
    printf("Results are correct.\n");
}

// ---------------- Main ----------------
int main() {
    srand((unsigned int)time(NULL));
    size_t size = N * sizeof(float);

    // Host arrays
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c_cpu = (float*)malloc(size);
    float *h_c_gpu = (float*)malloc(size);

    init_array(h_a, N);
    init_array(h_b, N);

    // ---------------- CPU benchmark ----------------
    double cpu_time_ms = benchmark_cpu(h_a, h_b, h_c_cpu, N, CPU_RUNS);
    printf("Average CPU time over %d runs: %.3f ms\n", CPU_RUNS, cpu_time_ms);

    // ---------------- GPU memory allocation ----------------
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // ---------------- GPU benchmark ----------------
    float gpu_time_ms = benchmark_gpu(d_a, d_b, d_c, N, GPU_RUNS);
    cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);
    printf("Average GPU kernel time over %d runs: %.3f ms\n", GPU_RUNS, gpu_time_ms);

    // ---------------- Check results ----------------
    check_result(h_c_cpu, h_c_gpu, N);

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);

    return 0;
}
