#include <iostream>
#include <cmath>
#include "utils.hpp"

// ---------------- Checking results ---------------- 
bool check_result(const float* cpu, const float* gpu, int n) {
    for (int i = 0; i < n; i++) {
        if (fabs(cpu[i] - gpu[i]) > 1e-6) {
            std::cerr << "Mismatch at index " << i
                      << ": CPU=" << cpu[i] << ", GPU=" << gpu[i] << std::endl;
            return false;
        }
    }
    std::cout << "Results match!" << std::endl;
    return true;
}

// ---------------- Initialize arrays ----------------
void init_array(float* a, int n) {
    for (int i = 0; i < n; i++) {
        a[i] = (float)(rand() % 1000) / 100.0f;  // values in [0, 10)
    }
}