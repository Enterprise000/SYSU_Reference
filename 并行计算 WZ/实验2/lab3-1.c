#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#define M 128
#define N 128
#define K 128


typedef struct {
	int value;
	int row;
	int col;
} element;

void times(element *A, element *B, element *C, int m, int n, int k, int row_per_prcs) {
    for (int i = 0; i < row_per_prcs; i++) {
        for (int j = 0; j < k; j++) {
            C[i * k + j].value = 0;
            for (int a = 0; a < n; a++) {
                C[i * k + j].value += A[i * n + a].value * B[a * k + j].value;  
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    int m = M,n = N,k = K;
    double start, end;

    element *A = NULL;
    element *B = NULL;
    element *C = NULL;
    element *Mem_A = NULL;  
    element *Mem_C = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    

	int row_per_prcs = m / size;

   Mem_A = (element *)malloc(row_per_prcs * n * sizeof(element));
    Mem_C = (element *)malloc(row_per_prcs * k * sizeof(element));
	//element *my_element = (element*)malloc(sizeof(element));

	element my_element;
	MPI_Datatype MPI_element;
	int block_lengths[3] = {1,1,1};
	MPI_Datatype types[3] = {MPI_INT,MPI_INT,MPI_INT};
	MPI_Aint value_address, row_address, col_address;
	MPI_Get_address(&my_element.value,&value_address);
	MPI_Get_address(&my_element.row,&row_address);
	MPI_Get_address(&my_element.col,&col_address);
	MPI_Aint displacement[3] = {0,row_address-value_address,col_address-row_address};
	MPI_Type_create_struct(3,block_lengths,displacement,types,&MPI_element);
	MPI_Type_commit(&MPI_element);
	
    if (rank == 0) {
    	srand(time(NULL));
        A = (element *)malloc(m * n * sizeof(element));
        B = (element *)malloc(n * k * sizeof(element));
        C = (element *)malloc(m * k * sizeof(element));

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                A[i * n + j].value = rand() % 100;
            }
        }

	for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                B[i * k + j].value = rand() % 100;
            }
        }
    } else{
        A = (element *)malloc(row_per_prcs * n * sizeof(element));
        B = (element *)malloc(n * k * sizeof(element));
        C = (element *)malloc(row_per_prcs * k * sizeof(element));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    MPI_Scatter(A, row_per_prcs * n, MPI_element, Mem_A, row_per_prcs * n, MPI_element, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, n * k, MPI_element, 0, MPI_COMM_WORLD);

    times(Mem_A, B, Mem_C, m,n,k, row_per_prcs);
    
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gather(Mem_C, row_per_prcs * k, MPI_element, C, row_per_prcs * k, MPI_element, 0, MPI_COMM_WORLD);

    end = MPI_Wtime();

    if (rank == 0) {
        printf("size %d x %d and %d x %d using np %d time is %f prc_sec\n", m, n, n, k, size, end - start);
        free(C);  
    }

    free(Mem_A);
    free(Mem_C);
    if (rank == 0) {
        free(A);
        free(B);
    }

    MPI_Finalize();
    return 0;
}
