# include<iostream>
# include<cstdlib>
# include<ctime>
# include<cuda_runtime.h>

// __global__ void  multiply(double* A, double* B, double* C, int m, int n, int k) {
    // int row = blockIdx.y * blockDim.y + threadIdx.y;
    // int col = blockIdx.x * blockDim.x + threadIdx.x;

    // if (row < m && col < k) {
       //  double value = 0;
       //  for (int i = 0; i < n; ++i) {
       //     value += A[row * n + i] * B[i * k + col];
       //  }
      //  C[row * k + col] = value;
  //  }
// }

//__global__ void multiply(double* A, double* B, double* C, int m, int n, int k) {
//    int row = blockIdx.y * blockDim.y + threadIdx.y;
//    int col = blockIdx.x * blockDim.x + threadIdx.x;
//
//    int blockSize = 2;
//
//    for (int i = 0; i < blockSize; ++i) {
//        for (int j = 0; j < blockSize; ++j) {
//            int blockRow = row + i;
//            int blockCol = col + j;
//
//            if (blockRow < m && blockCol < k) {
//                double value = 0;
//                for (int l = 0; l < n; ++l) {
//                    value += A[blockRow * n + l] * B[l * k + blockCol];
//                }
//                C[blockRow * k + blockCol] = value;
//            }
//        }
//    }
//}

__global__ void multiply(double* A, double* B, double* C, int m, int n, int k) {
	
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ double shared_A[32][32];  
    __shared__ double shared_B[32][32];

    double value = 0;
    
    for (int t = 0; t < (n + 32 - 1) / 32; t++) {
        if (row < m && t * 32 + threadIdx.x < n) {
            shared_A[threadIdx.y][threadIdx.x] = A[row * n + t * 32 + threadIdx.x];
        } else {
            shared_A[threadIdx.y][threadIdx.x] = 0;
        }
        if (col < k && t * 32 + threadIdx.y < n) {
            shared_B[threadIdx.y][threadIdx.x] = B[(t * 32 + threadIdx.y) * k + col];
        } else {
            shared_B[threadIdx.y][threadIdx.x] = 0;
        }
        
        __syncthreads(); 

        for (int i = 0; i < 32; i++) {
            value += shared_A[threadIdx.y][i] * shared_B[i][threadIdx.x];
        }

        __syncthreads(); 
    }

    if (row < m && col < k) {
        C[row * k + col] = value;
    }
}

int main(int argc, char *argv[]) {
	//     ??    ? 
	int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);
	
    if (m < 128 || m > 2048 || n < 128 || n > 2048 || k < 128 || k > 2048) {
        std::cerr << "Invalid Size" << std::endl;
        return -1;
    }
	
    double* A = new double[m * n];
    double* B = new double[n * k];
    double* C = new double[m * k];
	
    srand(time(0));
    for (int i = 0; i < m * n; ++i) {
        A[i] = static_cast<double>(rand()) / RAND_MAX * 1000.0;
    }
    for (int i = 0; i < n * k; ++i) {
        B[i] = static_cast<double>(rand()) / RAND_MAX * 1000.0;
    }

    size_t sizeA = m * n * sizeof(double);
    size_t sizeB = n * k * sizeof(double);
    size_t sizeC = m * k * sizeof(double);

    cudaError_t err1;
    cudaError_t err2;
    cudaError_t err3;
    
    double *d_A, *d_B, *d_C;

    err1 = cudaMalloc(&d_A, sizeA);
    err2 = cudaMalloc(&d_B, sizeB);
    err3 = cudaMalloc(&d_C, sizeC);
    
    if (err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess) {
        std::cerr << "cudaMalloc ?  : " << cudaGetErrorString(err1) << std::endl;
        return -1;
    }
    
    cudaMemcpy(d_A, A, sizeA,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB,cudaMemcpyHostToDevice);
    
    //      ?  ??  §³ 
    dim3 threadsPerBlock(32,32);
    dim3 numBlocks((m + threadsPerBlock.x - 1) / threadsPerBlock.x, (k + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    multiply<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, m, n, k);
    cudaMemcpy(C, d_C, sizeC,cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout<<"Matrix A: "<<std::endl;
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            std::cout << A[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout<<"Matrix B: "<<std::endl;
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            std::cout << B[i * k + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout<<"Result: "<<std::endl;
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            std::cout << C[i * k + j] << " ";
        }
        std::cout << std::endl;
    }   
      std::cout<<"Runtime: "<<milliseconds<<std::endl;
    std::cout << "   Enter    ?  ? g ?   " << std::endl;
    std::cin.get();

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] A;
    delete[] B;
    delete[] C;
    
    return 0;
}


