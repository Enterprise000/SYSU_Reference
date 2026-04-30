#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel function
__global__ void hello_world_kernel(int m, int n, int k) {
    printf("Hello World from thread (%d, %d) in block %d\n", threadIdx.x, threadIdx.y, blockIdx.x );
}

int main(int argc, char *argv[]) {
	int m = atoi(argv[1]);
	int n = atoi(argv[2]);
	int k = atoi(argv[3]);
	
    // Number of threads in each thread block
    dim3 threadsPerBlock(m,k);

    // Number of thread blocks in the grid
    dim3 blocksInGrid(n,1);

    // Launch the kernel
    hello_world_kernel<<<blocksInGrid, threadsPerBlock>>>(m,n,k);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
   
      printf("Hello world from the host!\n");
    // Check for any errors launching the kernel
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "hello_world_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
