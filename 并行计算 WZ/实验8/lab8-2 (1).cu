# include<iostream>
# include<cstdlib>
# include<ctime>
# include<cuda_runtime.h>

/*__global__ void transpose(double* A, double * AT, int n){
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row < n && col < n){
		AT[col * n + row] = A[row * n + col];
	}
}*/

__global__ void transpose(double* A, double * AT, int n){
	__shared__  double temp[16][16];
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row < n && col < n){
		temp[threadIdx.x][threadIdx.y] = A[row * n + col];
	}
	__syncthreads();
	if(row < n && col < n){
		AT[col*n+row] = temp[threadIdx.x][threadIdx.y];
	}
}


int main(int argc, char *argv[]) {
	int n = atoi(argv[1]);
	
	if(n < 512 || n > 2048){
		std::cerr<<"Invalid Size"<<std::endl;
		return -1;
	}
	
   double* A = new double[n * n];
	double* AT = new double[n * n];
	
	for (int i = 0; i < n * n; ++i) {
        A[i] = static_cast<double>(rand()) / RAND_MAX * 1000.0f;  //    ֵ  [0, 100]֮  
    }

   size_t allocationSize = n * n * sizeof(double);

    cudaError_t err1;
    cudaError_t err2;
  double *d_ptr = nullptr;
  double *dt_ptr =nullptr;

    //    GPU  Ϸ    ڴ 
    err1 = cudaMalloc(&d_ptr, allocationSize);
    err2 = cudaMalloc(&dt_ptr, allocationSize);
    if (err1 != cudaSuccess) {
        std::cerr << "cudaMalloc ʧ  : " << cudaGetErrorString(err1) << std::endl;
        return -1;
    }
    if (err2 != cudaSuccess) {
        std::cerr << "cudaMalloc ʧ  : " << cudaGetErrorString(err2) << std::endl;
        return -1;
    }

    std::cout << "   GPU  Ϸ      " << allocationSize / (1024 * 1024) << " MB  ڴ档" << std::endl;
    
    cudaMemcpy(d_ptr, A, allocationSize,cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(8,8);
    dim3 numBlocks((n+threadsPerBlock.x-1) / (threadsPerBlock.x), (n+threadsPerBlock.y-1)/threadsPerBlock.y);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    transpose<<<numBlocks, threadsPerBlock>>>(d_ptr,dt_ptr,n);
    cudaMemcpy(AT, dt_ptr, allocationSize,cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout<<"Runtime: "<<milliseconds<<std::endl;
    std::cout<<"Origin Matrix"<<std::endl;
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            std::cout << A[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout<<"Transposed Matrix"<<std::endl;
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            std::cout << AT[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
    //    ֳ        Ա    GPU  ڴ    
    std::cout << "   Enter    ͷ  ڴ沢 ˳   " << std::endl;
    std::cin.get();

    //  ͷŷ     ڴ 
    err1= cudaFree(d_ptr);
    if (err1 != cudaSuccess) {
        std::cerr << "cudaFree ʧ  : " << cudaGetErrorString(err1) << std::endl;
        return -1;
    }

    std::cout << " ڴ    ͷš " << std::endl;
    return 0;
}
