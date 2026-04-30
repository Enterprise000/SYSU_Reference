//使用Ptheads实现并行矩阵乘法
//输入m,n,k
//对AB进行矩阵乘法运算
//输出矩阵运算用时
//使用Pthreads创建多线程实现并行矩阵乘法
//线程数量1-16，矩阵规模128-2048 

#include<stdio.h>
#include<stdlib.h>
#include <pthread.h>
#include<time.h>
#include <mkl.h> 
#include <math.h>

typedef struct{
	int m;
	int n;
	int k;
	double *A,*B,*C;
	int id;
	int thread;
}matrix;

void *multiply(void *args){
    matrix *params = (matrix *)args;
    int m = params->m;
    int n = params->n;
    int k = params->k;
    double *A = params->A;
    double *B = params->B;
    double *C = params->C;
    int id = params->id;
    int thread = params->thread;
    
    //获得计算起止位置 
    int row_per_thread = m / thread;
    int start_row = id * row_per_thread;
    int end_row = (id==thread-1)?m:(id+1)*row_per_thread;

	//分块计算 
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < k; j++) {
            C[i * k + j] = 0;
            for (int l = 0; l < n; l++) {
                C[i * k + j] += A[i * n + l] * B[l * k + j];
            }
        }
    }
    pthread_exit(NULL);	
}

void verify(double *C, double *C_mkl, int m, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            if (fabs(C[i * k + j] - C_mkl[i * k + j]) > 1e-5) {
                printf("Error: Results do not match at C[%d][%d], result for pthread is %f, result for mkl is %f.\n", i, j,C[i * k + j],C_mkl[i * k + j]);
                return;
            }
        }
    }
    printf("Results correct.\n");
}

int main(int argc, char *argv[]){
//	获取m,n,k 
	int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);
    int threads = atoi(argv[4]);
    
//  分配内存    
	double *A = (double *)malloc(m * n * sizeof(double));
	double *B = (double *)malloc(n * k * sizeof(double));
	double *C = (double *)malloc(m * k * sizeof(double));
	double *C_mkl = (double *)malloc(m * k * sizeof(double));	
//    初始化 
    srand(time(NULL));
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                A[i * n + j] = rand() % 1000;
            }
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                B[i * n + j] = rand() % 1000;
            }
        }
    
//	创建线程
	pthread_t *thread = (pthread_t *)malloc(threads * sizeof(pthread_t));
	matrix *mtx = (matrix *)malloc(threads * sizeof(matrix));
	
	clock_t start = clock();
	int row_per_thread = m / threads;
	for(int i=0; i<threads;i++){
		int start_row = i * row_per_thread;
		int end_row = (i==threads-1)?m:(i+1)*row_per_thread; 
		mtx[i].m = m;
		mtx[i].n = n;
		mtx[i].k = k;
		mtx[i].A = A;
		mtx[i].B = B;
		mtx[i].C = C;
		mtx[i].thread = threads;
		pthread_create(&thread[i],NULL,multiply,&mtx[i]);
	}
	
	//等待所有线程结束 
for(int i = 0; i < threads; i++) {
        pthread_join(thread[i], NULL);
    }
    clock_t end = clock();
    
    //计算运行时间并输出 
    double runtime = ((double)(end-start))/CLOCKS_PER_SEC;
    printf("Matrix Size: %d, Threads: %d, Runtime: %f seconds\n",m,threads,runtime);

    //验证计算
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, k, n, 1.0, A, n, B, k, 0.0, C_mkl, k);
	verify(&C, &C_mkl, m, k);    
	free(A);
    free(B);
    free(C);
	free(thread);
	free(mtx);

    return 0;
}
