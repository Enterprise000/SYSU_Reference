#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstring>
#include <cuda_runtime.h>
#include <cmath>

#define CHANNELS 3
#define KERNEL_SIZE 3
#define KERNEL_NUM 3

// read ppm
bool read_ppm(const std::string& filename, std::vector<unsigned char>& image, int& width, int& height) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) return false;

    std::string magic;
    file >> magic;
    if (magic != "P6") return false;

    file.ignore(); 
    file >> width >> height;
    int max_val;
    file >> max_val;
    file.ignore(); 

    image.resize(width * height * 3);
    file.read(reinterpret_cast<char*>(image.data()), image.size());
    return true;
}

// save ppm
void write_ppm(const std::string& filename, const std::vector<float>& data, int width, int height) {
    std::ofstream file(filename, std::ios::binary);
    file << "P6\n" << width << " " << height << "\n255\n";
    for (int i = 0; i < width * height * 3; ++i) {
        float val = fminf(fmaxf(data[i], 0.0f), 1.0f);
        file.put(static_cast<unsigned char>(val * 255));
    }
}

__global__ void conv2d_kernel(const float* input, const float* kernels, float* output,
                              int H, int W, int outH, int outW, int stride, int padding) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z;

    if (out_x >= outW || out_y >= outH || k >= KERNEL_NUM) return;

    float sum = 0;
    for (int c = 0; c < CHANNELS; ++c) {
        for (int dy = 0; dy < KERNEL_SIZE; ++dy) {
            for (int dx = 0; dx < KERNEL_SIZE; ++dx) {
                int in_y = out_y * stride + dy - padding;
                int in_x = out_x * stride + dx - padding;
                float val = 0.0f;
                if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W)
                    val = input[(c * H + in_y) * W + in_x];
                float kernel_val = kernels[((k * CHANNELS + c) * KERNEL_SIZE + dy) * KERNEL_SIZE + dx];
                sum += val * kernel_val;
            }
        }
    }
    output[(k * outH + out_y) * outW + out_x] = sum;
}

/*
__global__ void conv2d_kernel(const float* input, const float* kernels, float* output,
                              int H, int W, int outH, int outW, int stride, int padding) {
    // 2*2 each thread
    int block_size = 2;  
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z;
    if (out_x >= outW || out_y >= outH || k >= KERNEL_NUM) return;

    for (int dy = 0; dy < block_size; ++dy) {
        for (int dx = 0; dx < block_size; ++dx) {
            int pixel_x = out_x + dx;
            int pixel_y = out_y + dy;

            if (pixel_x < outW && pixel_y < outH) {
                float sum = 0.0f;
                for (int c = 0; c < CHANNELS; ++c) {
                    for (int ky = 0; ky < KERNEL_SIZE; ++ky) {
                        for (int kx = 0; kx < KERNEL_SIZE; ++kx) {
                            int in_y = pixel_y * stride + ky - padding;
                            int in_x = pixel_x * stride + kx - padding;
                            float val = 0.0f;
                            if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W)
                                val = input[(c * H + in_y) * W + in_x];
                            float kernel_val = kernels[((k * CHANNELS + c) * KERNEL_SIZE + ky) * KERNEL_SIZE + kx];
                            sum += val * kernel_val;
                        }
                    }
                }
                output[(k * outH + pixel_y) * outW + pixel_x] = sum;
            }
        }
    }
}
*/

/*__global__ void conv2d_kernel_shared(
    const float* input, const float* kernels, float* output,
    int H, int W, int outH, int outW, int stride, int padding)
{
    const int smem_width = blockDim.x * stride + 2;
    const int smem_height = blockDim.y * stride + 2;

    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z; 

    extern __shared__ float smem[]; 
    int channel_block_size = smem_width * smem_height;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    for (int c = 0; c < CHANNELS; ++c) {
        for (int y = ty; y < smem_height; y += blockDim.y) {
            for (int x = tx; x < smem_width; x += blockDim.x) {
                int in_x = blockIdx.x * blockDim.x * stride + x - padding;
                int in_y = blockIdx.y * blockDim.y * stride + y - padding;
                float val = 0.f;
                if (in_x >= 0 && in_x < W && in_y >= 0 && in_y < H) {
                    val = input[(c * H + in_y) * W + in_x];
                }
                smem[c * channel_block_size + y * smem_width + x] = val;
            }
        }
    }

    __syncthreads();

    if (out_x >= outW || out_y >= outH || k >= KERNEL_NUM) return;

    float sum = 0.f;
    for (int c = 0; c < CHANNELS; ++c) {
        for (int dy = 0; dy < KERNEL_SIZE; ++dy) {
            for (int dx = 0; dx < KERNEL_SIZE; ++dx) {
                int smem_x = tx * stride + dx;
                int smem_y = ty * stride + dy;
                float val = smem[c * channel_block_size + smem_y * smem_width + smem_x];
                float kernel_val = kernels[((k * CHANNELS + c) * KERNEL_SIZE + dy) * KERNEL_SIZE + dx];
                sum += val * kernel_val;
            }
        }
    }

    output[(k * outH + out_y) * outW + out_x] = sum;
}*/

void normalize_output(std::vector<float>& output, int size) {
    float min_val = output[0], max_val = output[0];
    for (float v : output) {
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
    }
    float range = max_val - min_val + 1e-6f;
    for (auto& v : output) v = (v - min_val) / range;
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
        for (int c = 0; c < 3; ++c)
            input[c * img_size + i] = img[i * 3 + c] / 255;

    float h_kernel[KERNEL_NUM * CHANNELS * KERNEL_SIZE * KERNEL_SIZE];
    for (int i = 0; i < KERNEL_NUM * CHANNELS * KERNEL_SIZE * KERNEL_SIZE; ++i)
        h_kernel[i] = 1;

    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, sizeof(float) * input.size());
    cudaMalloc(&d_kernel, sizeof(h_kernel));

    cudaMemcpy(d_input, input.data(), sizeof(float) * input.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice);

    for (int stride : {1, 2, 3}) {
        int padding = (KERNEL_SIZE - 1 - (stride - 1)) / 2 + 1;
        int outH = height - 2;
        int outW = width - 2;

        std::vector<float> h_output(KERNEL_NUM * outH * outW);
        cudaMalloc(&d_output, sizeof(float) * h_output.size());

        dim3 block(16, 16);
        dim3 grid((outW + 15) / 16, (outH + 15) / 16, KERNEL_NUM);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        conv2d_kernel<<<grid, block>>>(d_input, d_kernel, d_output,
                                       height, width, outH, outW,
                                       stride, padding);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        std::cout << "Stride = " << stride << ", Time = " << ms << " ms\n";

        cudaMemcpy(h_output.data(), d_output, sizeof(float) * h_output.size(), cudaMemcpyDeviceToHost);

        normalize_output(h_output, h_output.size());

        std::vector<float> output_img(outH * outW * 3);
        for (int i = 0; i < outH * outW; ++i)
            for (int c = 0; c < 3; ++c)
                output_img[i * 3 + c] = h_output[(c * outH + i / outW) * outW + (i % outW)];

        std::ostringstream fname;
        fname << "output_stride" << stride << ".ppm";
        write_ppm(fname.str(), output_img, outW, outH);

        cudaFree(d_output);
    }

    cudaFree(d_input);
    cudaFree(d_kernel);
    return 0;
}

