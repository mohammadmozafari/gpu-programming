sudo docker run --gpus all -it -v "$(pwd)":/app nvidia/cuda:12.0.1-cudnn8-devel-ubuntu22.04 bash

nvcc 01_vector_add.cu -o 01_vector_add