#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>
#include <chrono>

#ifdef __CUDACC__
#include <cuda_runtime.h>


#define CUDA_CHECK(call) \
    do { \
        cudaError_t _e = (call); \
        if (_e != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s (%d)\n", __FILE__, __LINE__, cudaGetErrorString(_e), _e); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

static uint32_t swap32(uint32_t v) {
    return (v>>24) | ((v>>8)&0xFF00) | ((v<<8)&0xFF0000) | (v<<24);
}

static std::vector<uint8_t> read_idx(const std::string& path, int& n, int& r, int& c) {
    printf("[INFO] Reading IDX: %s\n", path.c_str());
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "[ERROR] Failed to open %s\n", path.c_str()); exit(EXIT_FAILURE); }
    uint32_t magic, num_items, rows=1, cols=1;
    fread(&magic, 4, 1, f);
    magic = swap32(magic);
    fread(&num_items, 4, 1, f);
    n = num_items = swap32(num_items);

    if (magic == 0x00000803) {
        fread(&rows, 4, 1, f);
        r = rows = swap32(rows);
        fread(&cols, 4, 1, f);
        c = cols = swap32(cols);
        printf("  -> Images: Count=%d, Dim=%dx%d\n", n, r, c);
    } else if (magic == 0x00000801) {
        r = c = 1;
        printf("  -> Labels: Count=%d\n", n);
    } else {
        fprintf(stderr, "[ERROR] Invalid magic number 0x%08X in %s\n", magic, path.c_str());
        fclose(f); exit(EXIT_FAILURE);
    }

    size_t data_size = (size_t)n * r * c;
    std::vector<uint8_t> data(data_size);
    size_t bytes_read = fread(data.data(), 1, data_size, f);
    if (bytes_read != data_size) { fprintf(stderr, "[ERROR] Incomplete read from %s\n", path.c_str()); fclose(f); exit(EXIT_FAILURE); }
    fclose(f);
    return data;
}

__global__ void relu_kernel(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = fmaxf(0.f, x[i]);
    }
}

__global__ void maxpool2x2_kernel(const float* __restrict__ x, float* y, int N, int C, int H, int W) {
    int H_out = H / 2;
    int W_out = W / 2;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tot = N * C * H_out * W_out;
    if (idx >= tot) return;

    int w2 = idx % W_out;
    int h2 = (idx / W_out) % H_out;
    int c  = (idx / (W_out * H_out)) % C;
    int n  = idx / (C * H_out * W_out);

    int h0 = h2 * 2, w0 = w2 * 2;

    const float* x_offset = x + (n * C + c) * H * W;
    float m = x_offset[h0 * W + w0];
    m = fmaxf(m, x_offset[(h0+1) * W + w0]);
    m = fmaxf(m, x_offset[h0 * W + w0+1]);
    m = fmaxf(m, x_offset[(h0+1) * W + w0+1]);
    y[idx] = m;
}

__global__ void conv3x3_kernel(const float* __restrict__ x,
                               const float* __restrict__ w,
                               const float* __restrict__ b,
                               float*       __restrict__ y,
                               int N, int H, int W, int Cin, int Cout) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tot = N * Cout * H * W;
    if (idx >= tot) return;

    int w0 = idx % W;
    int h0 = (idx / W) % H;
    int co = (idx / (H * W)) % Cout;
    int n  = idx / (Cout * H * W);

    float acc = b[co];

    for (int ci = 0; ci < Cin; ++ci) {
        const float* w_offset = w + (co * Cin + ci) * 3 * 3;
        const float* x_offset = x + (n * Cin + ci) * H * W;

        for (int dr = -1; dr <= 1; ++dr) {
            for (int dc = -1; dc <= 1; ++dc) {
                int h = h0 + dr;
                int c = w0 + dc;

                if (h >= 0 && h < H && c >= 0 && c < W) {
                    int x_idx = h * W + c;
                    int w_idx = (dr+1) * 3 + (dc+1);
                    acc += x_offset[x_idx] * w_offset[w_idx];
                }
            }
        }
    }
    y[idx] = acc;
}

__global__ void fc_kernel(const float* __restrict__ x, 
                          const float* __restrict__ w, 
                          const float* __restrict__ b, 
                          float*       __restrict__ y, 
                          int N, int D, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tot = N * K;
    if (idx >= tot) return;

    int n = idx / K;
    int k = idx % K;

    float acc = b[k];

    for (int d = 0; d < D; ++d) {
        acc += x[n*D + d] * w[k*D + d];
    }
    y[idx] = acc;
}

struct GpuNet {
    static constexpr int H0=28, W0=28, C0=1;
    static constexpr int C1=8;
    static constexpr int H1=28, W1=28, H1p=14, W1p=14;
    static constexpr int FC_IN = H1p*W1p*C1;
    static constexpr int FC_OUT=10;

    int N;

    float *d_x, *d_c1w, *d_c1b, *d_y1, *d_y1p,
          *d_fc_w, *d_fc_b, *d_fc_out;

    GpuNet(int nImg): N(nImg) {
        printf("[INFO] Initializing GpuNet (N=%d)...\n", N);

        auto mal=[&](float** p, size_t s, const char* name){
            CUDA_CHECK(cudaMalloc(p, s*sizeof(float)));
        };

        mal(&d_x,     (size_t)N*C0*H0*W0, "d_x");
        mal(&d_c1w,   (size_t)C1*3*3*C0,  "d_c1w"); mal(&d_c1b, C1, "d_c1b");
        mal(&d_y1,    (size_t)N*C1*H1*W1, "d_y1");  mal(&d_y1p, (size_t)N*C1*H1p*W1p, "d_y1p");
        mal(&d_fc_w,  (size_t)FC_OUT*FC_IN, "d_fc_w"); mal(&d_fc_b, FC_OUT, "d_fc_b");
        mal(&d_fc_out,(size_t)N*FC_OUT,   "d_fc_out");
        printf("[INFO] GPU memory allocated.\n");

        auto load_weights = [&](const char* prefix, char suffix, float* d_ptr, size_t num_elements) {
            char fpath[256];
            snprintf(fpath, sizeof(fpath), "%s_nchw.%c", prefix, suffix);
            printf("  Loading weights: %s (%zu elements)...\n", fpath, num_elements);
            FILE *f = fopen(fpath, "rb");
            if (!f) { fprintf(stderr, "[ERROR] Failed to open weight file %s\n", fpath); exit(EXIT_FAILURE); }
            std::vector<float> h_buf(num_elements);
            size_t read_count = fread(h_buf.data(), sizeof(float), num_elements, f);
            fclose(f);
            if (read_count != num_elements) { fprintf(stderr, "[ERROR] Incomplete read from %s\n", fpath); exit(EXIT_FAILURE); }
            CUDA_CHECK(cudaMemcpy(d_ptr, h_buf.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice));
        };

        load_weights("conv1", 'w', d_c1w, C1*C0*3*3);
        load_weights("conv1", 'b', d_c1b, C1);
        load_weights("fc",    'w', d_fc_w, FC_OUT*FC_IN);
        load_weights("fc",    'b', d_fc_b, FC_OUT);
        printf("[INFO] Weights loaded.\n");
    }

    ~GpuNet() {
        printf("[INFO] Freeing GPU memory...\n");
        cudaFree(d_x); cudaFree(d_c1w); cudaFree(d_c1b); cudaFree(d_y1);
        cudaFree(d_y1p); cudaFree(d_fc_w); cudaFree(d_fc_b); cudaFree(d_fc_out);
        printf("[INFO] GPU memory freed.\n");
    }

    void forward(const uint8_t* img) {
        std::vector<float> h_x(N * C0 * H0 * W0);
        for (size_t i = 0; i < h_x.size(); ++i) {
            h_x[i] = img[i] / 255.0f;
        }
        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice));

        int threads = 256;
        auto grid = [&](int total_elems) { return (total_elems + threads - 1) / threads; };

        conv3x3_kernel<<<grid(N*C1*H1*W1), threads>>>(
            d_x, d_c1w, d_c1b, d_y1, N, H1, W1, C0, C1);
        relu_kernel<<<grid(N*C1*H1*W1), threads>>>(d_y1, N*C1*H1*W1);
        maxpool2x2_kernel<<<grid(N*C1*H1p*W1p), threads>>>(
            d_y1, d_y1p, N, C1, H1, W1);

        fc_kernel<<<grid(N*FC_OUT), threads>>>(
            d_y1p, d_fc_w, d_fc_b, d_fc_out, N, FC_IN, FC_OUT);
    }

    int accuracy(const uint8_t* img, const uint8_t* lbl) {
        printf("[INFO] Running forward pass for accuracy calculation...\n");
        forward(img);

        std::vector<float> h_fc_out(N * FC_OUT);
        CUDA_CHECK(cudaMemcpy(h_fc_out.data(), d_fc_out, h_fc_out.size() * sizeof(float), cudaMemcpyDeviceToHost));

        int correct = 0;
        for (int i = 0; i < N; ++i) {
            float max_val = -1e9;
            int pred = -1;
            for (int j = 0; j < FC_OUT; ++j) {
                if (h_fc_out[i * FC_OUT + j] > max_val) {
                    max_val = h_fc_out[i * FC_OUT + j];
                    pred = j;
                }
            }
            if (pred == lbl[i]) {
                correct++;
            }
        }
        printf("[INFO] Accuracy calculation complete.\n");
        return correct;
    }
};

static void run_gpu() {
    int n_test, h_test, w_test, n_lbl, r_lbl, c_lbl;
    std::vector<uint8_t> test_img = read_idx("t10k-images-idx3-ubyte", n_test, h_test, w_test);
    std::vector<uint8_t> test_lbl = read_idx("t10k-labels-idx1-ubyte", n_lbl, r_lbl, c_lbl);
    if (n_test != n_lbl) { fprintf(stderr, "[ERROR] Mismatch between image count (%d) and label count (%d)\n", n_test, n_lbl); exit(EXIT_FAILURE); }
    printf("[INFO] MNIST test data loaded: %d images, %d labels.\n", n_test, n_lbl);

    GpuNet net(n_test);

    printf("[INFO] Benchmarking GPU inference...\n");
    double sec = 0.0;
    int num_runs = 10;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_runs; ++i) {
        net.forward(test_img.data());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    sec = duration.count() / num_runs;
    printf("[INFO] Benchmark complete.\n");

    int top1 = net.accuracy(test_img.data(), test_lbl.data());

    double throughput = n_test / sec;
    double acc_percent = top1 * 100.0 / n_test;
    std::cout << "----------------------------------------\n";
    std::cout << "GPU Result:\n";
    std::cout << "  Throughput: " << throughput << " img/s\n";
    std::cout << "  Accuracy:   " << acc_percent << "% (" << top1 << "/" << n_test << ")\n";
    std::cout << "----------------------------------------\n";
}

int main() {
    printf("========== MNIST GPU Inference ==========" "\n");
    run_gpu();
    printf("===============================================\n");
    return 0;
}

#else // __CUDACC__ not defined
int main() {
    fprintf(stderr, "[ERROR] This program requires CUDA and must be compiled with nvcc.\n");
    return 1;
}
#endif // __CUDACC__