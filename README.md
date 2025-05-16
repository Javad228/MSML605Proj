# MNIST TinyCNN Profiling and Analysis

## Quick Start

For the easiest setup, simply open the provided Jupyter notebook in [Google Colab](https://colab.research.google.com/) and run all cells—no local installation required.

## Manual Setup (Optional)

1. 
    ```bash
    pip install -r requirements.txt
    ```
2. 
    ```bash
    nvcc -O3 -o mnist_gpu gpu_mnist.cu
    nvcc -O3 -o mnist_gpu_nchw gpu_mnist_nchw.cu
    ```
3. 
    ```bash
    bash detailed_profiling_fixed.sh
    ```