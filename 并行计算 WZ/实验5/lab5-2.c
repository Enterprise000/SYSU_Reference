# include<stdio.h>
# include<stdlib.h>
# include<time.h>
# include<omp.h>

void* matrix_multiply(int* A, int* B, int* C, int m, int n,int k){
	# pragma omp parallel for schedule(guided,2)
	for(int i=0; i<m; i++){
		for(int j=0; j<k;j++){
			C[i * k + j] = 0;
			for(int a=0; a<n;a++){
				C[i*k+j]+= A[i*n+a] * B[a*k+j];
			}
		}
	}
}

int main(int argc, char *argv[]){
	int thread_num = atoi(argv[4]);
	int m = atoi(argv[1]);
	int n = atoi(argv[2]);
	int k = atoi(argv[3]);
	
//	prepare matrix
    int* A = (int *)malloc(m * n * sizeof(int));
    int* B = (int *)malloc(n * k * sizeof(int));
    int* C = (int *)malloc(m * k * sizeof(int));
    
    srand(time(NULL));
    for (int i = 0; i < m * n; i++) {
        A[i] = rand() % 1000;
    }
    for (int i = 0; i < n * k; i++) {
        B[i] = rand() % 1000;
    }
	
    omp_set_num_threads(thread_num);
    double start_time = omp_get_wtime();
    matrix_multiply(A,B,C,m,n,k);
    double end_time = omp_get_wtime();
    double runtime = end_time - start_time;
    printf("Matrix A: %d * %d, Matrix B: %d * %d, Threads: %d, Runtime: %.4f seconds\n",m,n,n,k,thread_num,runtime);

    printf("matrix A: \n");
    for(int i=0;i<m; i++){
	for(int j=0; j<n; j++){
	  printf("%d ",A[i*n+j]);
}
	printf("\n");
}

    printf("matrix B: \n");
    for(int i=0;i<n; i++){
	for(int j=0; j<k; j++){
	  printf("%d ",A[i*k+j]);
}
	printf("\n");
}

    printf("matrix C: \n");
    for(int i=0;i<m; i++){
	for(int j=0; j<k; j++){
	  printf("%d ",A[i*k+j]);
}
	printf("\n");
}
	free(A);
	free(B);
	free(C);
	
	return 0;
    
}
