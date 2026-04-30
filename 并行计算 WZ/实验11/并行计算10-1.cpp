#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHANNELS 3
#define KERNEL_SIZE 3
#define KERNEL_NUM 3

bool read_ppm(const std::string& filename, std::vector<unsigned char>& image, int& width, int& height) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) return false;

    std::string magic;
    file >> magic;
    if (magic != "P6") return false;

    file >> width >> height;
    int max_val;
    file >> max_val;
    file.get(); 

    image.resize(width * height * 3);
    file.read(reinterpret_cast<char*>(image.data()), image.size());
    return true;
}

bool write_ppm(const std::string& filename, const std::vector<float>& image, int width, int height) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) return false;

    file << "P6\n" << width << " " << height << "\n255\n";
    for (float v : image) {
        unsigned char pixel = static_cast<unsigned char>(std::min(std::max(v, 0.0f), 1.0f) * 255.0f);
        file.write(reinterpret_cast<char*>(&pixel), 1);
    }
    return true;
}

int get_output_dim(int in_size, int kernel_size, int stride, int padding) {
    return (in_size + 2 * padding - kernel_size) / stride + 1;
}

void normalize_output(std::vector<float>& output) {
    float min_val = output[0], max_val = output[0];
    for (float v : output) {
        min_val = std::min(min_val, v);
        max_val = std::max(max_val, v);
    }
    float range = max_val - min_val + 1e-6f;
    for (float& v : output) v = (v - min_val) / range;
}

// im2col CUDA kernel
/*__global__ void im2col_kernel(const float* input, float* columns,
                              int C, int H, int W,
                              int KH, int KW,
                              int outH, int outW,
                              int stride, int padding) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int total_cols = outH * outW;

    if (col >= total_cols) return;

    int out_y = col / outW;
    int out_x = col % outW;

    int col_offset = 0;
    for (int c = 0; c < C; ++c) {
        for (int ky = 0; ky < KH; ++ky) {
            for (int kx = 0; kx < KW; ++kx) {
                int in_y = out_y * stride + ky - padding;
                int in_x = out_x * stride + kx - padding;
                float val = 0.0f;
                if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W)
                    val = input[(c * H + in_y) * W + in_x];
                columns[col * (C * KH * KW) + col_offset++] = val;
            }
        }
    }
}*/

__global__ void im2col_kernel(const float* input, float* columns,
                                       int C, int H, int W,
                                       int KH, int KW,
                                       int outH, int outW,
                                       int stride, int padding) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_cols = outH * outW;

    for (int i = 0; i < 4; ++i) {
        int col = tid * 4 + i;
        if (col >= total_cols) return;

        int out_y = col / outW;
        int out_x = col % outW;

        int col_offset = 0;
        for (int c = 0; c < C; ++c) {
            for (int ky = 0; ky < KH; ++ky) {
                for (int kx = 0; kx < KW; ++kx) {
                    int in_y = out_y * stride + ky - padding;
                    int in_x = out_x * stride + kx - padding;
                    float val = 0.0f;
                    if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W)
                        val = input[(c * H + in_y) * W + in_x];
                    columns[col * (C * KH * KW) + col_offset++] = val;
                }
            }
        }
    }
}

__global__ void im2col_shared_kernel(const float* input, float* columns,
                                     int C, int H, int W,
                                     int KH, int KW,
                                     int outH, int outW,
                                     int stride, int padding) {
    int col = blockIdx.x;
    if (col >= outH * outW) return;

    int out_y = col / outW;
    int out_x = col % outW;

    const int patch_size = C * KH * KW;

    extern __shared__ float tile[]; 

    int tid = threadIdx.x;

    for (int i = tid; i < patch_size; i += blockDim.x) {
        int c = i / (KH * KW);
        int ky = (i / KW) % KH;
        int kx = i % KW;

        int in_y = out_y * stride + ky - padding;
        int in_x = out_x * stride + kx - padding;

        float val = 0.0f;
        if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W)
            val = input[(c * H + in_y) * W + in_x];

        tile[i] = val;
    }

    __syncthreads();

    if (tid == 0) {
        for (int i = 0; i < patch_size; ++i) {
            columns[i * (outH * outW) + col] = tile[i];
        }
    }
}

void run_im2col_convolution(const float* d_input, const float* d_kernel,
                            float* d_output, float* d_columns,
                            int C, int H, int W,
                            int KH, int KW, int K,
                            int stride, int padding,
                            int outH, int outW) {
    int col_size = C * KH * KW;
    int num_output = outH * outW;

    dim3 block(1024);
    dim3 grid((num_output + block.x - 1) / block.x);
    size_t shared_mem_size = sizeof(float) * C * KH * KW;
    im2col_kernel<<<grid, block>>>(d_input, d_columns, C, H, W, KH, KW, outH, outW, stride, padding);
    //im2col_shared_kernel<<<grid, block, shared_mem_size>>>(d_input, d_columns, C, H, W, KH, KW, outH, outW, stride, padding);

    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                K, num_output, col_size,
                &alpha,
                d_kernel, K,
                d_columns, col_size,
                &beta,
                d_output, K);

    cublasDestroy(handle);
}

int main() {
    std::string input_file = "image.ppm";
    std::vector<unsigned char> img;
    int width, height;
    if (!read_ppm(input_file, img, width, height)) {
        std::cerr << "can not find ppm file!\n";
        return -1;
    }

    int img_size = width * height;
    std::vector<float> input(CHANNELS * img_size);
    for (int i = 0; i < img_size; ++i)
        for (int c = 0; c < CHANNELS; ++c)
            input[c * img_size + i] = img[i * 3 + c] / 255;

    float h_kernel[KERNEL_NUM * CHANNELS * KERNEL_SIZE * KERNEL_SIZE];
    for (int i = 0; i < KERNEL_NUM * CHANNELS * KERNEL_SIZE * KERNEL_SIZE; ++i)
        h_kernel[i] = 1;

    float *d_input, *d_kernel;
    cudaMalloc(&d_input, sizeof(float) * input.size());
    cudaMalloc(&d_kernel, sizeof(h_kernel));
    cudaMemcpy(d_input, input.data(), sizeof(float) * input.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice);

    for (int stride : {1, 2, 3}) {
        int padding = (KERNEL_SIZE - 1 - (stride - 1)) / 2 + 1;
        int outH = get_output_dim(height, KERNEL_SIZE, stride, padding);
        int outW = get_output_dim(width, KERNEL_SIZE, stride, padding);

        std::vector<float> h_output(KERNEL_NUM * outH * outW);
        float *d_output, *d_columns;

        cudaMalloc(&d_output, sizeof(float) * h_output.size());
        cudaMalloc(&d_columns, sizeof(float) * CHANNELS * KERNEL_SIZE * KERNEL_SIZE * outH * outW);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        run_im2col_convolution(d_input, d_kernel, d_output, d_columns,
                               CHANNELS, height, width,
                               KERNEL_SIZE, KERNEL_SIZE, KERNEL_NUM,
                               stride, padding, outH, outW);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        std::cout << "Stride = " << stride << ", Time = " << ms << " ms\n";

        cudaMemcpy(h_output.data(), d_output, sizeof(float) * h_output.size(), cudaMemcpyDeviceToHost);
        normalize_output(h_output);

for (int k = 0; k < KERNEL_NUM; ++k) {
    std::vector<float> output_img(outH * outW * 3);
    for (int i = 0; i < outH; ++i) {
        for (int j = 0; j < outW; ++j) {
            int idx = i * outW + j;
            float val = h_output[k * outH * outW + idx];
            for (int c = 0; c < 3; ++c)
                output_img[(i * outW + j) * 3 + c] = val;
        }
    }
    std::ostringstream fname;
    fname << "output_stride" << stride << "_kernel" << k << ".ppm";
    write_ppm(fname.str(), output_img, outW, outH);
}

        cudaFree(d_output);
        cudaFree(d_columns);
    }

    cudaFree(d_input);
    cudaFree(d_kernel);
    return 0;
}
