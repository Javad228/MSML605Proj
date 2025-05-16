#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>

#ifdef __CUDACC__
#include <cuda_runtime.h>

#define CUDA_ASSERT(cmd) \
    do { \
        cudaError_t _err = (cmd); \
        if (_err != cudaSuccess) { \
            fprintf(stderr, "CUDA failure at %s:%d: %s (%d)\n", __FILE__, __LINE__, cudaGetErrorString(_err), _err); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

static uint32_t reverse32(uint32_t val) {
    return (val>>24) | ((val>>8)&0xFF00) | ((val<<8)&0xFF0000) | (val<<24);
}

static std::vector<uint8_t> load_idx(const std::string& fname, int& count, int& rows, int& cols) {
    printf("[INFO] Loading IDX: %s\n", fname.c_str());
    FILE* file = fopen(fname.c_str(), "rb");
    if (!file) { fprintf(stderr, "[ERR] Cannot open %s\n", fname.c_str()); exit(EXIT_FAILURE); }
    uint32_t magic, n, h=1, w=1;
    fread(&magic, 4, 1, file);
    printf("Magic (pre): 0x%08X\n", magic);
    magic = reverse32(magic);
    printf("Magic (post): 0x%08X\n", magic);
    fread(&n, 4, 1, file);
    count = n = reverse32(n);
    if (magic == 0x00000803) {
        fread(&h, 4, 1, file);
        rows = h = reverse32(h);
        fread(&w, 4, 1, file);
        cols = w = reverse32(w);
        printf("  -> Images: %d, %dx%d\n", count, rows, cols);
    } else if (magic == 0x00000801) {
        rows = cols = 1;
        printf("  -> Labels: %d\n", count);
    } else {
        fprintf(stderr, "[ERR] Bad magic 0x%08X in %s\n", magic, fname.c_str());
        fclose(file); exit(EXIT_FAILURE);
    }
    size_t sz = (size_t)count * rows * cols;
    std::vector<uint8_t> buf(sz);
    size_t got = fread(buf.data(), 1, sz, file);
    fclose(file);
    return buf;
}

__global__ void act_relu(float* arr, int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        arr[idx] = fmaxf(0.f, arr[idx]);
    }
}

__global__ void pool2x2(const float* __restrict__ src, float* dst, int B, int H, int W, int D) {
    int outH = H / 2;
    int outW = W / 2;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * outH * outW * D;
    if (tid >= total) return;
    int d  = tid % D;
    int x2 = (tid / D) % outW;
    int y2 = (tid / D / outW) % outH;
    int b  = tid / (D * outH * outW);
    int y0 = y2 * 2, x0 = x2 * 2;
    float mx = src[((b*H + y0)*W + x0)*D + d];
    mx = fmaxf(mx, src[((b*H + y0+1)*W + x0)*D + d]);
    mx = fmaxf(mx, src[((b*H + y0)*W + x0+1)*D + d]);
    mx = fmaxf(mx, src[((b*H + y0+1)*W + x0+1)*D + d]);
    dst[tid] = mx;
}

__global__ void conv3x3(const float* __restrict__ inp,
                        const float* __restrict__ filt,
                        const float* __restrict__ bias,
                        float*       __restrict__ out,
                        int B, int H, int W, int Din, int Dout) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * W * Dout;
    if (tid >= total) return;
    int od = tid % Dout;
    int ox = (tid / Dout) % W;
    int oy = (tid / Dout / W) % H;
    int ob = tid / (Dout * H * W);
    float acc = bias[od];
    for (int id = 0; id < Din; ++id) {
        #pragma unroll
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int y = oy + dy;
                int x = ox + dx;
                if (y >= 0 && y < H && x >= 0 && x < W) {
                    int inp_idx = ((ob*H + y)*W + x)*Din + id;
                    int filt_idx = ((od*3 + (dy+1))*3 + (dx+1))*Din + id;
                    acc += inp[inp_idx] * filt[filt_idx];
                }
            }
        }
    }
    out[tid] = acc;
}

__global__ void dense_layer(const float* __restrict__ inp,
                           const float* __restrict__ wt,
                           const float* __restrict__ bias,
                           float*       __restrict__ out,
                           int B, int Din, int Dout) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * Dout;
    if (tid >= total) return;
    int b = tid / Dout;
    int o = tid % Dout;
    float acc = bias[o];
    for (int i = 0; i < Din; ++i) {
        acc += inp[b*Din + i] * wt[o*Din + i];
    }
    out[tid] = acc;
}

struct DeviceNet {
    static constexpr int IN_H=28, IN_W=28, IN_C=1;
    static constexpr int CONV_C=8;
    static constexpr int CONV_H=28, CONV_W=28, POOL_H=14, POOL_W=14;
    static constexpr int FCIN = POOL_H*POOL_W*CONV_C;
    static constexpr int FCOUT=10;

    int batch;
    float *g_img, *g_conv_w, *g_conv_b, *g_conv_out, *g_pool,
          *g_fc_w, *g_fc_b, *g_fc_out;

    DeviceNet(int n): batch(n) {
        printf("[INFO] Allocating DeviceNet (batch=%d)...\n", batch);
        auto alloc=[&](float** ptr, size_t sz, const char* label){
            CUDA_ASSERT(cudaMalloc(ptr, sz*sizeof(float)));
        };
        alloc(&g_img,     (size_t)batch*IN_H*IN_W*IN_C, "g_img");
        alloc(&g_conv_w,  (size_t)CONV_C*3*3*IN_C,  "g_conv_w"); alloc(&g_conv_b, CONV_C, "g_conv_b");
        alloc(&g_conv_out,(size_t)batch*CONV_H*CONV_W*CONV_C, "g_conv_out");  alloc(&g_pool, (size_t)batch*POOL_H*POOL_W*CONV_C, "g_pool");
        alloc(&g_fc_w,    (size_t)FCOUT*FCIN, "g_fc_w"); alloc(&g_fc_b, FCOUT, "g_fc_b");
        alloc(&g_fc_out,  (size_t)batch*FCOUT,   "g_fc_out");
        printf("[INFO] CUDA memory ready.\n");
        auto loadw = [&](const char* fname, float* dptr, size_t len){
            printf("  Loading '%s' (%zu floats)...\n", fname, len);
            std::vector<float> buf(len);
            FILE* file = fopen(fname, "rb");
            if (!file) { fprintf(stderr, "[ERR] Cannot open '%s'\n", fname); exit(EXIT_FAILURE); }
            size_t got = fread(buf.data(), sizeof(float), len, file);
            fclose(file);
            if (got != len) { fprintf(stderr, "[ERR] Size mismatch for '%s'. Expected %zu, got %zu\n", fname, len, got); exit(EXIT_FAILURE); }
            CUDA_ASSERT(cudaMemcpy(dptr, buf.data(), len*sizeof(float), cudaMemcpyHostToDevice));
        };
        printf("[INFO] Loading model params...\n");
        loadw("conv1.w", g_conv_w, (size_t)CONV_C*3*3*IN_C);
        loadw("conv1.b", g_conv_b, (size_t)CONV_C);
        loadw("fc.w",    g_fc_w, (size_t)FCOUT*FCIN);
        loadw("fc.b",    g_fc_b, (size_t)FCOUT);
        printf("[INFO] Model params loaded.\n");
    }
    ~DeviceNet() {
        printf("[INFO] Releasing CUDA memory...\n");
        cudaFree(g_img);
        cudaFree(g_conv_w);
        cudaFree(g_conv_b);
        cudaFree(g_conv_out);
        cudaFree(g_pool);
        cudaFree(g_fc_w);
        cudaFree(g_fc_b);
        cudaFree(g_fc_out);
    }
    void propagate(const uint8_t* img) {
        std::vector<float> h_img((size_t)batch*IN_H*IN_W*IN_C);
        for (size_t i = 0; i < h_img.size(); ++i) h_img[i] = img[i] / 255.f;
        CUDA_ASSERT(cudaMemcpy(g_img, h_img.data(), h_img.size()*sizeof(float), cudaMemcpyHostToDevice));
        int blk = 256;
        auto grid = [&](int n) { return dim3((n + blk - 1) / blk); };
        int conv_sz = batch*CONV_H*CONV_W*CONV_C;
        conv3x3<<<grid(conv_sz), blk>>>(g_img, g_conv_w, g_conv_b, g_conv_out, batch, IN_H, IN_W, IN_C, CONV_C);
        act_relu<<<grid(conv_sz), blk>>>(g_conv_out, conv_sz);
        int pool_sz = batch*POOL_H*POOL_W*CONV_C;
        pool2x2<<<grid(pool_sz), blk>>>(g_conv_out, g_pool, batch, CONV_H, CONV_W, CONV_C);
        int fc_sz = batch*FCOUT;
        dense_layer<<<grid(fc_sz), blk>>>(g_pool, g_fc_w, g_fc_b, g_fc_out, batch, FCIN, FCOUT);
        CUDA_ASSERT(cudaGetLastError());
    }
    double timeit(const uint8_t* img) {
        cudaEvent_t st,en; CUDA_ASSERT(cudaEventCreate(&st)); CUDA_ASSERT(cudaEventCreate(&en));
        CUDA_ASSERT(cudaEventRecord(st));
        propagate(img);
        CUDA_ASSERT(cudaEventRecord(en));
        CUDA_ASSERT(cudaEventSynchronize(en));
        float ms; CUDA_ASSERT(cudaEventElapsedTime(&ms, st, en));
        CUDA_ASSERT(cudaEventDestroy(st)); CUDA_ASSERT(cudaEventDestroy(en));
        return ms/1000.0;
    }
    int assess(const uint8_t* img, const uint8_t* lbl) {
        printf("[INFO] Running forward for accuracy...\n");
        propagate(img);
        std::vector<float> h_out((size_t)batch*FCOUT);
        printf("  Copying %zu logits from device...\n", h_out.size());
        CUDA_ASSERT(cudaMemcpy(h_out.data(), g_fc_out, h_out.size()*sizeof(float), cudaMemcpyDeviceToHost));
        int nright = 0;
        for (int i = 0; i < batch; ++i) {
            int pred = 0; float maxv = -1e9f;
            for (int j = 0; j < FCOUT; ++j) {
                float v = h_out[i*FCOUT + j];
                if (v > maxv) { maxv = v; pred = j; }
            }
            if (pred == lbl[i]) ++nright;
        }
        printf("[INFO] Accuracy done.\n");
        return nright;
    }
};

static void launch_gpu() {
    int nImg, rImg, cImg, nLbl, rLbl, cLbl;
    auto imgs = load_idx("t10k-images-idx3-ubyte", nImg, rImg, cImg);
    auto lbls = load_idx("t10k-labels-idx1-ubyte", nLbl, rLbl, cLbl);
    if (nImg != 10000 || nLbl != 10000 || nImg != nLbl) {
        fprintf(stderr, "[ERR] MNIST test set mismatch (expected 10000).\n");
        return;
    }
    printf("[INFO] MNIST test loaded: %d imgs, %d lbls.\n", nImg, nLbl);
    DeviceNet model(nImg);
    printf("[INFO] Benchmarking device inference...\n");
    double sec = model.timeit(imgs.data());
    printf("[INFO] Benchmark done.\n");
    int top1 = model.assess(imgs.data(), lbls.data());
    double throughput = nImg / sec;
    double acc_percent = top1 * 100.0 / nImg;
    std::cout << "========================================\n";
    std::cout << "Device Result:\n";
    std::cout << "  Throughput: " << throughput << " img/s\n";
    std::cout << "  Accuracy:   " << acc_percent << "% (" << top1 << "/" << nImg << ")\n";
    std::cout << "========================================\n";
}

int main() {
    printf("========== MNIST GPU Run==========""\n");
    launch_gpu();
    printf("=============================================\n");
    return 0;
}

#else
#error "This file must be compiled with nvcc (NVIDIA CUDA Compiler)"
#endif