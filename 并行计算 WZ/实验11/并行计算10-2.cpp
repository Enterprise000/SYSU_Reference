#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cudnn.h>
#include <cuda_runtime.h>

#define CHANNELS 3
#define KERNEL_SIZE 3
#define KERNEL_NUM 3

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

void write_ppm(const std::string& filename, const std::vector<float>& data, int width, int height) {
    std::ofstream file(filename, std::ios::binary);
    file << "P6\n" << width << " " << height << "\n255\n";
    for (int i = 0; i < width * height * 3; ++i) {
        float val = fminf(fmaxf(data[i], 0.0f), 1.0f);
        file.put(static_cast<unsigned char>(val * 255));
    }
}

void normalize_output(std::vector<float>& output) {
    float min_val = *std::min_element(output.begin(), output.end());
    float max_val = *std::max_element(output.begin(), output.end());
    float range = max_val - min_val + 1e-6f;
    for (auto& v : output) v = (v - min_val) / range;
}

int main() {
  std::string input_file = "image.ppm";
    // image vector
    std::vector<unsigned char> img;
    int width, height;
    // load input image(.ppm)
    if (!read_ppm(input_file, img, width, height)) {
        std::cerr << "can not find ppm file\n";
        return -1;
    }
    
 	//pixel vector(standard)
    int imgSize = width * height;
    std::vector<float> input(CHANNELS * imgSize);
    for (int i = 0; i < imgSize; ++i)
        for (int c = 0; c < 3; ++c)
            input[c * imgSize + i] = img[i * 3 + c] / 255;

    // kernel array(kernelnum * kernelsize * kernelsize * channels)
    float laplacian_kernel[KERNEL_SIZE * KERNEL_SIZE] = {
     	0,  1,  0,
     	1, -4,  1,
     	0,  1,  0
	};
    float kernel[KERNEL_NUM * CHANNELS * KERNEL_SIZE * KERNEL_SIZE];
    for (int k = 0; k < KERNEL_NUM; ++k) {
    for (int c = 0; c < CHANNELS; ++c) {
        for (int i = 0; i < KERNEL_SIZE * KERNEL_SIZE; ++i) {
            kernel[k * CHANNELS * KERNEL_SIZE * KERNEL_SIZE + c * KERNEL_SIZE * KERNEL_SIZE + i] = laplacian_kernel[i];
        }
    }
	}
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, sizeof(float) * input.size());
    cudaMalloc(&d_kernel, sizeof(kernel));
    cudaMemcpy(d_input, input.data(), sizeof(float) * input.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, sizeof(kernel), cudaMemcpyHostToDevice);

    for (int stride : {1, 2, 3}) {
    	// input descriptor
        cudnnTensorDescriptor_t input_desc, output_desc;
        cudnnFilterDescriptor_t filter_desc;
        cudnnConvolutionDescriptor_t conv_desc;
        cudnnCreateTensorDescriptor(&input_desc);
        cudnnCreateFilterDescriptor(&filter_desc);
        cudnnCreateConvolutionDescriptor(&conv_desc);
        int pad = (KERNEL_SIZE - 1) / 2;
        cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                   1, CHANNELS, height, width);
        cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                   KERNEL_NUM, CHANNELS, KERNEL_SIZE, KERNEL_SIZE);
        cudnnSetConvolution2dDescriptor(conv_desc, pad, pad, stride, stride,
                                        1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

        int n, c, h, w;
        cudnnGetConvolution2dForwardOutputDim(conv_desc, input_desc, filter_desc, &n, &c, &h, &w);
        cudnnCreateTensorDescriptor(&output_desc);
        cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
        size_t workspace_size = 0;
        cudnnConvolutionFwdAlgo_t algo;
        cudnnGetConvolutionForwardAlgorithm(cudnn, input_desc, filter_desc, conv_desc, output_desc,
                                            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo);
        cudnnGetConvolutionForwardWorkspaceSize(cudnn, input_desc, filter_desc, conv_desc,
                                                output_desc, algo, &workspace_size);
        void* d_workspace;
        cudaMalloc(&d_workspace, workspace_size);
        cudaMalloc(&d_output, sizeof(float) * n * c * h * w);

        float alpha = 1, beta = 0;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        cudnnConvolutionForward(cudnn, &alpha,
                                input_desc, d_input,
                                filter_desc, d_kernel,
                                conv_desc, algo,
                                d_workspace, workspace_size,
                                &beta,
                                output_desc, d_output);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        std::cout << "Stride = " << stride << ", Time = " << ms << " ms\n";

        std::vector<float> h_output(KERNEL_NUM * h * w);
        cudaMemcpy(h_output.data(), d_output, sizeof(float) * h_output.size(), cudaMemcpyDeviceToHost);
        normalize_output(h_output);

        std::vector<float> rgb_output(h * w * 3);
        for (int i = 0; i < h * w; ++i)
            for (int c = 0; c < 3; ++c)
                rgb_output[i * 3 + c] = h_output[(c * h + i / w) * w + (i % w)];

        std::ostringstream fname;
        fname << "output_stride" << stride << ".ppm";
        write_ppm(fname.str(), rgb_output, w, h);

        cudaFree(d_output);
        cudaFree(d_workspace);
        cudnnDestroyTensorDescriptor(output_desc);
        cudnnDestroyConvolutionDescriptor(conv_desc);
        cudnnDestroyFilterDescriptor(filter_desc);
        cudnnDestroyTensorDescriptor(input_desc);
    }

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudnnDestroy(cudnn);
    return 0;
}

