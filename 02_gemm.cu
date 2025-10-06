// 02_gemm.cu
// Single-file benchmark: CPU naive GEMM vs cuBLAS SGEMM
// Usage: ./gemm_bench [M N K num_runs]
// Default: M=N=K=1024, num_runs=10
//
// Compile: nvcc -O3 -lcublas -o gemm_bench gemm_bench.cu

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <random>
#include <iostream>
#include <iomanip>

#include <cuda_runtime.h>
#include <cublas_v2.h>

static void checkCuda(cudaError_t err, const char *msg = nullptr) {
    if (err != cudaSuccess) {
        if (msg) fprintf(stderr, "%s: ", msg);
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

static void checkCublas(cublasStatus_t s, const char *msg = nullptr) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        if (msg) fprintf(stderr, "%s: ", msg);
        fprintf(stderr, "cuBLAS error %d\n", (int)s);
        std::exit(EXIT_FAILURE);
    }
}

// Indexing for column-major arrays: element (row, col) -> col*rows + row
inline size_t idx_colmajor(size_t row, size_t col, size_t rows) {
    return col * rows + row;
}

// Allocate host array (column-major)
float* host_alloc_mat_colmajor(size_t rows, size_t cols) {
    float* ptr = nullptr;
    // pinned host memory can improve transfer performance
    checkCuda(cudaMallocHost((void**)&ptr, rows * cols * sizeof(float)), "cudaMallocHost");
    return ptr;
}

void host_free_mat(float* ptr) {
    if (ptr) cudaFreeHost(ptr);
}

void fill_random_colmajor(float* A, size_t rows, size_t cols, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    size_t total = rows * cols;
    for (size_t j = 0; j < cols; ++j) {
        for (size_t i = 0; i < rows; ++i) {
            A[idx_colmajor(i, j, rows)] = dist(rng);
        }
    }
}

void zero_mat_colmajor(float* A, size_t rows, size_t cols) {
    size_t total = rows * cols;
    for (size_t i = 0; i < total; ++i) A[i] = 0.0f;
}

// CPU naive GEMM on column-major data
// C = alpha * A * B + beta * C
void cpu_gemm_colmajor(size_t M, size_t N, size_t K,
                       float alpha,
                       const float* A, // M x K
                       const float* B, // K x N
                       float beta,
                       float* C)       // M x N
{
    // Triple loop: i (row), j (col), k (summation)
    // Access pattern uses column-major indexing: idx_colmajor(row, col, M)
    for (size_t j = 0; j < N; ++j) {
        for (size_t i = 0; i < M; ++i) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                float a_ik = A[idx_colmajor(i, k, M)]; // A(i,k)
                float b_kj = B[idx_colmajor(k, j, K)]; // B(k,j)
                sum += a_ik * b_kj;
            }
            C[idx_colmajor(i, j, M)] = alpha * sum + beta * C[idx_colmajor(i, j, M)];
        }
    }
}

// Run cuBLAS SGEMM: C = alpha * A * B + beta * C
// All matrices are in column-major layout on device.
void gpu_gemm_cublas(cublasHandle_t handle,
                     size_t M, size_t N, size_t K,
                     float alpha,
                     const float* d_A, // device ptr M x K
                     const float* d_B, // device ptr K x N
                     float beta,
                     float* d_C)       // device ptr M x N
{
    // Leading dimensions for column-major layout:
    int lda = static_cast<int>(M);
    int ldb = static_cast<int>(K);
    int ldc = static_cast<int>(M);

    // cublasSgemm: C = alpha * A * B + beta * C
    checkCublas(
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                    &alpha,
                    d_A, lda,
                    d_B, ldb,
                    &beta,
                    d_C, ldc),
        "cublasSgemm failed");
}

double measure_cpu_gemm_avg_gflops(size_t M, size_t N, size_t K,
                                   const float* A, const float* B,
                                   float* C, int runs)
{
    // Warmup run
    zero_mat_colmajor(C, M, N);
    cpu_gemm_colmajor(M, N, K, 1.0f, A, B, 0.0f, C);

    std::vector<double> times;
    times.reserve(runs);
    for (int r = 0; r < runs; ++r) {
        zero_mat_colmajor(C, M, N);
        auto t0 = std::chrono::high_resolution_clock::now();
        cpu_gemm_colmajor(M, N, K, 1.0f, A, B, 0.0f, C);
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = t1 - t0;
        times.push_back(elapsed.count());
        printf("CPU run %2d: time = %.6f s, GFLOPS = %.3f\n",
               r+1, elapsed.count(), (2.0 * M * N * K) / (elapsed.count() * 1e9));
    }
    double avg = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    double avg_gflops = (2.0 * M * N * K) / (avg * 1e9);
    printf("CPU average: time = %.6f s, GFLOPS = %.3f\n", avg, avg_gflops);
    return avg_gflops;
}

double measure_gpu_gemm_avg_gflops(size_t M, size_t N, size_t K,
                                   cublasHandle_t handle,
                                   const float* d_A, const float* d_B, float* d_C,
                                   int runs)
{
    // We'll use CUDA events to time the GPU kernel (including cublas call).
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "cudaEventCreate start");
    checkCuda(cudaEventCreate(&stop), "cudaEventCreate stop");

    // Warmup (one call to populate any lazy context)
    checkCuda(cudaEventRecord(start), "cudaEventRecord warmup start");
    gpu_gemm_cublas(handle, M, N, K, 1.0f, d_A, d_B, 0.0f, d_C);
    checkCuda(cudaEventRecord(stop), "cudaEventRecord warmup stop");
    checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize warmup");

    std::vector<float> times_ms;
    times_ms.reserve(runs);
    for (int r = 0; r < runs; ++r) {
        checkCuda(cudaEventRecord(start), "cudaEventRecord start");
        gpu_gemm_cublas(handle, M, N, K, 1.0f, d_A, d_B, 0.0f, d_C);
        checkCuda(cudaEventRecord(stop), "cudaEventRecord stop");
        checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize");
        float ms = 0.0f;
        checkCuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
        times_ms.push_back(ms);
        double seconds = ms / 1000.0;
        printf("GPU run %2d: time = %.3f ms, GFLOPS = %.3f\n",
               r+1, ms, (2.0 * M * N * K) / (seconds * 1e9));
    }
    double avg_ms = std::accumulate(times_ms.begin(), times_ms.end(), 0.0f) / times_ms.size();
    double avg_gflops = (2.0 * M * N * K) / ((avg_ms / 1000.0) * 1e9);
    printf("GPU average: time = %.3f ms, GFLOPS = %.3f\n", avg_ms, avg_gflops);

    checkCuda(cudaEventDestroy(start), "cudaEventDestroy start");
    checkCuda(cudaEventDestroy(stop), "cudaEventDestroy stop");
    return avg_gflops;
}

float max_abs_diff_colmajor(const float* A, const float* B, size_t M, size_t N) {
    float maxdiff = 0.0f;
    for (size_t j = 0; j < N; ++j) {
        for (size_t i = 0; i < M; ++i) {
            float a = A[idx_colmajor(i, j, M)];
            float b = B[idx_colmajor(i, j, M)];
            float diff = std::fabs(a - b);
            if (diff > maxdiff) maxdiff = diff;
        }
    }
    return maxdiff;
}

int main(int argc, char* argv[]) {
    size_t M = 1024, N = 1024, K = 1024;
    int num_runs = 10;

    if (argc >= 4) {
        M = static_cast<size_t>(std::stoul(argv[1]));
        N = static_cast<size_t>(std::stoul(argv[2]));
        K = static_cast<size_t>(std::stoul(argv[3]));
    }
    if (argc >= 5) {
        num_runs = std::stoi(argv[4]);
    }

    printf("GEMM benchmark (column-major layout)\n");
    printf("Matrix sizes: M=%zu, N=%zu, K=%zu\n", M, N, K);
    printf("Number of timed runs: %d\n", num_runs);

    // Print device info
    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable devices detected.\n");
        return EXIT_FAILURE;
    }
    int device = 0;
    checkCuda(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp devProp;
    checkCuda(cudaGetDeviceProperties(&devProp, device), "cudaGetDeviceProperties");
    printf("Using device %d: %s (compute %d.%d), totalGlobalMem=%.2f GB\n",
           device, devProp.name, devProp.major, devProp.minor, devProp.totalGlobalMem / (1024.0*1024.0*1024.0));
    printf("\n");

    // Allocate and initialize host matrices (column-major)
    float* h_A = host_alloc_mat_colmajor(M, K); // M x K
    float* h_B = host_alloc_mat_colmajor(K, N); // K x N
    float* h_C_cpu = host_alloc_mat_colmajor(M, N); // M x N
    float* h_C_gpu = host_alloc_mat_colmajor(M, N); // M x N (result copied back)

    fill_random_colmajor(h_A, M, K, 1337);
    fill_random_colmajor(h_B, K, N, 4242);
    zero_mat_colmajor(h_C_cpu, M, N);
    zero_mat_colmajor(h_C_gpu, M, N);

    // CPU benchmark
    printf("-------- CPU benchmark (naive triple loop) --------\n");
    double cpu_gflops = measure_cpu_gemm_avg_gflops(M, N, K, h_A, h_B, h_C_cpu, num_runs);
    printf("\n");

    // Setup cuBLAS and allocate device memory
    cublasHandle_t cublas_handle;
    checkCublas(cublasCreate(&cublas_handle), "cublasCreate");

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    checkCuda(cudaMalloc((void**)&d_A, bytes_A), "cudaMalloc d_A");
    checkCuda(cudaMalloc((void**)&d_B, bytes_B), "cudaMalloc d_B");
    checkCuda(cudaMalloc((void**)&d_C, bytes_C), "cudaMalloc d_C");

    // Copy host to device
    checkCuda(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice), "cudaMemcpy H2D A");
    checkCuda(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice), "cudaMemcpy H2D B");

    // GPU benchmark
    printf("-------- GPU benchmark (cuBLAS sgemm) --------\n");
    double gpu_gflops = measure_gpu_gemm_avg_gflops(M, N, K, cublas_handle, d_A, d_B, d_C, num_runs);
    printf("\n");

    // Copy result back
    checkCuda(cudaMemcpy(h_C_gpu, d_C, bytes_C, cudaMemcpyDeviceToHost), "cudaMemcpy D2H C");

    // Validate results
    float maxdiff = max_abs_diff_colmajor(h_C_cpu, h_C_gpu, M, N);
    printf("Max absolute difference between CPU and GPU results: %.6e\n", maxdiff);
    const float tolerance = 1e-3f + 1e-6f * static_cast<float>(K);
    if (maxdiff > tolerance) {
        printf("WARNING: difference (%.6e) is larger than tolerance (%.6e).\n", maxdiff, tolerance);
    } else {
        printf("Result validation PASSED (tol=%.6e).\n", tolerance);
    }
    printf("\n");

    // Summary
    printf("Summary:\n");
    printf("  CPU GFLOPS: %.3f\n", cpu_gflops);
    printf("  GPU GFLOPS: %.3f\n", gpu_gflops);
    printf("  Speedup (GPU/CPU): %.3fx\n", gpu_gflops / (cpu_gflops + 1e-12));

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(cublas_handle);

    host_free_mat(h_A);
    host_free_mat(h_B);
    host_free_mat(h_C_cpu);
    host_free_mat(h_C_gpu);

    return 0;
}
