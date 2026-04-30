#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

#define N 2048

void times(int *A, int *B, int *C, int n, int row_prcs) {
    for (int i = 0; i < row_prcs; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0;
            for (int k = 0; k < n; k++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];  
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size, n = N;
    double start, end;

    int *A = NULL;
    int *B = NULL;
    int *C = NULL;
    int *Mem_A = NULL;  
    int *Mem_C = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int row_prcs = n / size;

    Mem_A = (int *)malloc(row_prcs * n * sizeof(int));
    Mem_C = (int *)malloc(row_prcs * n * sizeof(int));

    if (rank == 0) {
        A = (int *)malloc(n * n * sizeof(int));
        B = (int *)malloc(n * n * sizeof(int));
        C = (int *)malloc(n * n * sizeof(int));

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A[i * n + j] = rand() % 100;
                B[i * n + j] = rand() % 100;
            }
        }
    } else{
        B = (int *)malloc(n * n * sizeof(int));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    MPI_Scatter(A, row_prcs * n, MPI_INT, Mem_A, row_prcs * n, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, n * n, MPI_INT, 0, MPI_COMM_WORLD);

    times(Mem_A, B, Mem_C, n, row_prcs);
   
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gather(Mem_C, row_prcs * n, MPI_INT, C, row_prcs * n, MPI_INT, 0, MPI_COMM_WORLD);

    end = MPI_Wtime();

    if (rank == 0) {
        printf("size %d x %d using np %d time is %f \n", n, n, size, end - start);
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
