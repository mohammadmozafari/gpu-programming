#include <iostream>
#include <cmath>
#include "utils.hpp"

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