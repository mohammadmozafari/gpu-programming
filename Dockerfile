FROM nvidia/cuda:12.0.1-cudnn8-devel-ubuntu22.04

WORKDIR /workspace
COPY . .

RUN apt-get update && apt-get install -y cmake make g++
RUN mkdir build && cd build && cmake .. && make -j$(nproc)

ENTRYPOINT ["/workspace/build/bin/bench_gemm"]