# include<stdlib.h>
# include <stdio.h>
# include <math.h>
# include <omp.h>
# include<pthread.h>
# include<time.h>

# define M 512
# define N 512
# define K 512

void* matrix_multiply(int* A, int* B, int* C, int m, int n,int k){
	# pragma omp parallel for schedule(static)
	for(int i=0; i<m; i++){
		for(int j=0; j<k;j++){
			C[i * k + j] = 0;
			for(int a=0; a<n;a++){
				C[i*k+j]+= A[i*n+a] * B[a*k+j];
			}
		}
	}
}

extern void parallel_for(int start, int end, int inc, void *(*functor)(int, void*), void *arg, int num_threads); 

typedef struct{
	int *A;
	int *B;
	int *C;
}MatrixArg; 

void* matrix_multiply1(int i,void* arg){
	MatrixArg *mtxarg = (MatrixArg*)arg;
    int *A = mtxarg->A;
    int *B = mtxarg->B;
    int *C = mtxarg->C;
    for(int j=0; j<K; j++){
    	C[i * K + j] = 0;
		for(int a=0; a<N;a++){
			C[i*K+j]+= A[i*N+a] * B[a*K+j];
		}
	}
}

int main(int argc, char *argv[]){
	int thread_num = atoi(argv[1]);
	
//	prepare matrix
    int* A = (int *)malloc(M * N * sizeof(int));
    int* B = (int *)malloc(N * K * sizeof(int));
    int* C = (int *)malloc(M * K * sizeof(int));
    int* C1 = (int *)malloc(M * K * sizeof(int));
    
    srand(time(NULL));
    for (int i = 0; i < M * N; i++) {
        A[i] = rand() % 1000;
    }
    for (int i = 0; i < N * K; i++) {
        B[i] = rand() % 1000;
    }
	
    omp_set_num_threads(thread_num);
    double start_time = omp_get_wtime();
    MatrixArg mtxarg = {.A = A, .B = B, .C = C1};
    parallel_for(0,M,1,matrix_multiply1, &mtxarg,thread_num);
    double end_time = omp_get_wtime();
    double runtime = end_time - start_time;
    matrix_multiply(A,B,C,M,N,K);
    for(int i=0; i<M;i++){
    	for(int j=0; j<K; j++){
    		if(C[i * K + j] != C1[i * K + j]){
    			printf("Incorrect!");
    			free(A);
				free(B);
				free(C);
    			return 0;
			}
		}
	}
    printf("Matrix A: %d * %d, Matrix B: %d * %d, Threads: %d, Runtime: %f seconds\n",M,N,N,K,thread_num,runtime);
    
	free(A);
	free(B);
	free(C);
	
	return 0;
    
}
