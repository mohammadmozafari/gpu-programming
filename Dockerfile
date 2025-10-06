FROM nvidia/cuda:12.0.1-cudnn8-devel-ubuntu22.04

# Install build tools first (only runs once unless Dockerfile changes above)
RUN apt-get update && apt-get install -y cmake make g++ && rm -rf /var/lib/apt/lists/*

# Set up workdir
WORKDIR /workspace

# Copy only CMake files and source files
COPY CMakeLists.txt ./
COPY benchmarks ./benchmarks
COPY include ./include
COPY src ./src

# Build the project (can cache partial builds if nothing changed)
RUN mkdir build
WORKDIR /workspace/build
RUN cmake .. && make -j$(nproc)

# Copy rest of your repo if needed (for runtime data etc.)
WORKDIR /workspace
COPY . .

ENTRYPOINT ["/workspace/build/bin/bench_gemm"]