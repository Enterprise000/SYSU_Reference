# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <omp.h>
# include<pthread.h>

# define M 500
# define N 500

extern void parallel_for(int start, int end, int inc, void *(*functor)(int, void*), void *arg, int num_threads);

typedef struct{
  double (*w)[N];
  double* mean;
}MeanArg;

typedef struct{
  double (*w)[N];
  double (*u)[N];
}CopyArg;

typedef struct{
  double (*w)[N];
  double (*u)[N];
  double* diff;
  pthread_mutex_t *mutex;
}DiffArg;


void* set_w_i(int i, void* arg){
	double (*w)[N] = (double (*)[N]) arg;
	w[i][0] = 100.0;
	w[i][N-1] = 100.0;
}

void* set_w_j(int j, void* arg){
	double (*w)[N] = (double (*)[N]) arg;
	w[M-1][j] = 100.0;
	w[0][j] = 0.0;
}

void* add_mean(int index, void* arg){
    MeanArg *marg = (MeanArg*)arg;
    double *mean = marg->mean;
    double (*w)[N] = marg->w;
    int i = index / 2;
    int j = index % 2;
    if (i == 0 || i == M - 1) {
        *mean += w[i][j];
    } else if (j == 0 || j == N - 1) {
        *mean += w[i][j];
    }
    return NULL; 
}

void* set_mean(int i, void* arg){
    MeanArg *marg = (MeanArg*)arg;
    double mean = *(marg->mean);
    double (*w)[N] = marg->w;
     for (int j = 1; j < N - 1; j++) {
          w[i][j] = mean;
    }
    return NULL;
}

void* copy(int i, void* arg) {
    CopyArg *carg = (CopyArg*)arg;
    double (*u)[N] = carg->u;
    double (*w)[N] = carg->w;
    for (int j = 0; j < N; j++) {
        u[i][j] = w[i][j];
    }
}

void* update(int i, void* arg) {
    CopyArg *carg = (CopyArg*)arg;
    double (*u)[N] = carg->u;
    double (*w)[N] = carg->w;
    for (int j = 1; j < N - 1; j++) {
       w[i][j] = ( u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1] ) / 4.0;
    }
}

void* diff_cmp(int i, void* arg) {
    DiffArg *darg = (DiffArg*)arg;
    double (*u)[N] = darg->u;
    double (*w)[N] = darg->w;
    double *diff = darg->diff;
    double my_diff = 0.0;
    for (int j = 1; j < N - 1; j++) {
    	if(my_diff < fabs(w[i][j] - u[i][j])){
	    my_diff = fabs(w[i][j] - u[i][j]);
	}
    }
    
    pthread_mutex_t *mutex = darg->mutex;
    pthread_mutex_lock(mutex);
    if (my_diff > *diff) {
        *diff = my_diff;
    }
    pthread_mutex_unlock(mutex);
}

int main ( int argc, char *argv[] )
{
  double diff;
  double epsilon = 0.001;
  int i;
  int iterations;
  int iterations_print;
  int j;
  double mean;
  double my_diff;
  double u[M][N];
  double w[M][N];
  double wtime;

  printf ( "\n" );
  printf ( "HEATED_PLATE_OPENMP\n" );
  printf ( "  C/OpenMP version\n" );
  printf ( "  A program to solve for the steady state temperature distribution\n" );
  printf ( "  over a rectangular plate.\n" );
  printf ( "\n" );
  printf ( "  Spatial grid of %d by %d points.\n", M, N );
  printf ( "  The iteration will be repeated until the change is <= %e\n", epsilon ); 
  printf ( "  Number of processors available = %d\n", omp_get_num_procs ( ) );
  printf ( "  Number of threads =              %d\n", omp_get_max_threads ( ) );

  mean = 0.0;
  
  int num_threads = omp_get_max_threads ( ) ;
  parallel_for(1, M - 1, 1, set_w_i, w, num_threads);
  parallel_for(0, N, 1, set_w_j, w, num_threads);
  MeanArg marg = {.w = w, .mean = &mean};  

  parallel_for(1, M - 1, 1, add_mean, &marg, num_threads);
  parallel_for(0, N, 1, add_mean, &marg, num_threads);

  mean = mean / ( double ) ( 2 * M + 2 * N - 4 );
  printf ( "\n" );
  printf ( "  MEAN = %f\n", mean );

  parallel_for(1, M - 1, 1, set_mean, &marg, num_threads);

  iterations = 0;
  iterations_print = 1;
  printf ( "\n" );
  printf ( " Iteration  Change\n" );
//printf("-3");
  printf ( "\n" );
//printf("-2");
  wtime = omp_get_wtime ( );
//printf("-1");
  diff = epsilon;

  CopyArg carg = {.w = w, .u = u};

  while ( epsilon <= diff )
  {
//printf("0");
  	parallel_for(0, M, 1, copy, &carg, num_threads);
        //printf("1");
	parallel_for(1, M - 1, 1, update, &carg, num_threads);
	//printf("2");
    diff = 0.0;
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    DiffArg darg = {.w = w, .u = u, .diff = &diff,.mutex = &mutex};
    parallel_for(1, M - 1, 1, diff_cmp, &darg, num_threads);
    pthread_mutex_destroy(&mutex);
/*# pragma omp parallel shared ( diff, u, w ) private ( i, j, my_diff )
    {
      my_diff = 0.0;
# pragma omp for
      for ( i = 1; i < M - 1; i++ )
      {
        for ( j = 1; j < N - 1; j++ )
        {
          if ( my_diff < fabs ( w[i][j] - u[i][j] ) )
          {
            my_diff = fabs ( w[i][j] - u[i][j] );
          }
        }
      }
# pragma omp critical
      {
        if ( diff < my_diff )
        {
          diff = my_diff;
        }
      }
    }*/
	//printf("3");
    /*iterations++;
    if ( iterations == iterations_print )
    {
      printf ( "  %8d  %f\n", iterations, diff );
      iterations_print = 2 * iterations_print;
    }*/
/*
  C and C++ cannot compute a maximum as a reduction operation.

  Therefore, we define a private variable MY_DIFF for each thread.
  Once they have all computed their values, we use a CRITICAL section
  to update DIFF.
*/


    iterations++;
    if ( iterations == iterations_print )
    {
      printf ( "  %8d  %f\n", iterations, diff );
      iterations_print = 2 * iterations_print;
    }
  } 
  wtime = omp_get_wtime ( ) - wtime;

  printf ( "\n" );
  printf ( "  %8d  %f\n", iterations, diff );
  printf ( "\n" );
  printf ( "  Error tolerance achieved.\n" );
  printf ( "  Wallclock time = %f\n", wtime );

  printf ( "\n" );
  printf ( "HEATED_PLATE_OPENMP:\n" );
  printf ( "  Normal end of execution.\n" );

  return 0;

# undef M
# undef N
}
