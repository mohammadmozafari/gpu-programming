#include <stdio.h>
#include <cuda_runtime.h>

void checkCudaErrors(cudaError_t err) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__global__ void hello_cuda() {
    printf("Hello from GPU!\n");
}

int main() {
    hello_cuda<<<1, 1>>>();
    // Check for errors after the kernel launch
    checkCudaErrors(cudaGetLastError());
    
    // Wait for the kernel to finish and check for any runtime errors
    checkCudaErrors(cudaDeviceSynchronize());
    fflush(stdout); // Add this line to flush the output buffer
    return 0;
}