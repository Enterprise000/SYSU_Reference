# include <cstdlib>
# include <iostream>
# include <iomanip>
# include <cmath>
# include <ctime>
# include <omp.h>
# include <mpi.h>

using namespace std;

int main ( int argc, char* argv[]);
void ccopy ( int n, double x[], double y[] );
void cfft2 ( int n, double x[], double y[], double w[], double sgn );
void cffti ( int n, double w[] );
double cpu_time ( void );
double ggl ( double *ds );
void step ( int n, int mj, double a[], double b[], double c[], double d[], 
double w[], double sgn );
void timestamp ( );


void cfft2 ( int n, double x[], double y[], double w[], double sgn ){
    int rank, size;
    int m, mj, tgle;
    
    //MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    m = (int)(log((double)n) / log(1.99));
    
    mj = 1;
    tgle = 1;

    int n_prc = n / size;  
    double *x_prc = (double *)malloc(2*n_prc * sizeof(double));
    double *y_prc = (double *)malloc(2*n_prc * sizeof(double));
    
    MPI_Scatter(x, 2*n_prc, MPI_DOUBLE, x_prc, 2*n_prc, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(y, 2*n_prc, MPI_DOUBLE, y_prc, 2*n_prc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    step(n_prc, mj, &x_prc[0*2+0], &x_prc[(n_prc/2)*2+0],&y_prc[0*2+0], &y_prc[mj*2+0], w, sgn);

    if (n == 2) {
        //free(x_prc);
        //free(y_prc);
        return;
    }

    for (int j = 0; j < m - 2; j++) {
        mj = mj * 2;

        if (tgle) {
            //MPI_Bcast(y_prc, n_prc, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            step(n_prc, mj, &y_prc[0*2+0],&y_prc[(n_prc/2)*2+0], &x_prc[0*2+0], &x_prc[mj*2+0], w, sgn);
            tgle = 0;
        } else {
            //MPI_Bcast(x_prc, n_prc, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            step(n_prc, mj, &x_prc[0*2+0],&x_prc[(n_prc/2)*2+0], &y_prc[0*2+0], &y_prc[mj*2+0], w, sgn);
            tgle = 1;
        }
    }

    if (tgle) {
        ccopy(n_prc, y_prc, x_prc);
    }

    mj = n / 2;
    step(n_prc, mj, &x_prc[0*2+0],&x_prc[(n_prc/2)*2+0], &y_prc[0*2+0], &y_prc[mj*2+0], w, sgn);

    MPI_Gather(x_prc, 2*n_prc, MPI_DOUBLE, x, 2*n_prc, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(y_prc, 2*n_prc, MPI_DOUBLE, y, 2*n_prc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    free(x_prc);
    free(y_prc);
}

void step ( int n, int mj, double a[], double b[], double c[], double d[], double w[], double sgn ){
  double ambr;
  double ambu;
  int j;
  int ja;
  int jb;
  int jc;
  int jd;
  int jw;
  int k;
  int lj;
  int mj2;
  double wjw[2];

  mj2 = 2 * mj;
  lj = n / mj2;

  for ( j = 0; j < lj; j++ ){
    jw = j * mj;
    ja  = jw;
    jb  = ja;
    jc  = j * mj2;
    jd  = jc;

    wjw[0] = w[jw*2+0]; 
    wjw[1] = w[jw*2+1];

    if ( sgn < 0.0 ) {
      wjw[1] = - wjw[1];
    }

    for ( k = 0; k < mj; k++ ){
      c[(jc+k)*2+0] = a[(ja+k)*2+0] + b[(jb+k)*2+0];
      c[(jc+k)*2+1] = a[(ja+k)*2+1] + b[(jb+k)*2+1];

      ambr = a[(ja+k)*2+0] - b[(jb+k)*2+0];
      ambu = a[(ja+k)*2+1] - b[(jb+k)*2+1];

      d[(jd+k)*2+0] = wjw[0] * ambr - wjw[1] * ambu;
      d[(jd+k)*2+1] = wjw[1] * ambr + wjw[0] * ambu;
    }
  }
  return;
}

int main (int argc, char *argv[]){
  double ctime;
  double ctime1;
  double ctime2;
  double error;
  int first;
  double flops;
  double fnm1;
  int i;
  int icase;
  int it;
  int ln2;
  double mflops;
  int n;
  int nits = 10000;
  static double seed;
  double sgn;
  double *w;
  double *x;
  double *y;
  double *z;
  double z0;
  double z1;

  // initialize mpi 
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  
  timestamp ( );
  cout << "\n";
  if(rank==0){
  	cout << "FFT_SERIAL\n";
  	cout << "  C++ version\n";
  	cout << "\n";
  	cout << "  Demonstrate an implementation of the Fast Fourier Transform\n";
  	cout << "  of a complex data vector.\n";
  }

//
//  Prepare for tests.
//
  if(rank == 0){
  	cout << "\n";
  	cout << "  Accuracy check:\n";
  	cout << "\n";
  	cout << "    FFT ( FFT ( X(1:N) ) ) == N * X(1:N)\n";
  	cout << "\n";
  	cout << "             N      NITS    Error         Time          Time/Call     MFLOPS\n";
  	cout << "\n";
  }


  seed  = 331.0;
  n = 1;
  
  // initialize data and standard 
  for ( ln2 = 1; ln2 <= 20; ln2++ ){
    n = 2 * n;
    w = new double[  n];
    x = new double[2*n];
    y = new double[2*n];
    z = new double[2*n];

    first = 1;

    for ( icase = 0; icase < 2; icase++ ){

      if ( first ){
      	if(rank == 0){
      		// initialize
      	    for ( i = 0; i < 2 * n; i = i + 2 ){
          		z0 = ggl ( &seed );
          		z1 = ggl ( &seed );
          		x[i] = z0;
          		z[i] = z0;
          		x[i+1] = z1;
          		z[i+1] = z1;
        	}
		}

      } 
      else{
        for ( i = 0; i < 2 * n; i = i + 2 ){
          z0 = 0.0;
          z1 = 0.0;
          x[i] = z0;
          z[i] = z0;
          x[i+1] = z1;
          z[i+1] = z1;
        }
      }
//  Initialize the sine and cosine tables.
      cffti ( n, w );
//  Transform forward, back 
      if ( first ){
        MPI_Barrier(MPI_COMM_WORLD);

        sgn = + 1.0;
        cfft2 ( n, x, y, w, sgn );
        MPI_Barrier(MPI_COMM_WORLD);
        sgn = - 1.0;
        cfft2 ( n, y, x, w, sgn );
        MPI_Barrier(MPI_COMM_WORLD);
		//  Calculate error
        fnm1 = 1.0 / ( double ) n;
        error = 0.0;
        for ( i = 0; i < 2 * n; i = i + 2 ){
          error = error 
          + pow ( z[i]   - fnm1 * x[i], 2 )
          + pow ( z[i+1] - fnm1 * x[i+1], 2 );
        }
        error = sqrt ( fnm1 * error );
        cout << "  " << setw(12) << n
             << "  " << setw(8) << nits
             << "  " << setw(12) << error;
        first = 0;
      }
      else{
        ctime1 = cpu_time ( );
        for ( it = 0; it < nits; it++ ){
        	MPI_Barrier(MPI_COMM_WORLD);
          	sgn = + 1.0;
          	cfft2 ( n, x, y, w, sgn );
          	MPI_Barrier(MPI_COMM_WORLD);

          	sgn = - 1.0;
         	cfft2 ( n, y, x, w, sgn );
         	MPI_Barrier(MPI_COMM_WORLD);
        }
        ctime2 = cpu_time ( );
        ctime = ctime2 - ctime1;

        flops = 2.0 * ( double ) nits * ( 5.0 * ( double ) n * ( double ) ln2 );

        mflops = flops / 1.0E+06 / ctime;
        
        if(rank == 0){
            cout << "  " << setw(12) << ctime
            << "  " << setw(12) << ctime / ( double ) ( 2 * nits )
            << "  " << setw(12) << mflops << "\n";	
		}
      }
    }
    if ( ( ln2 % 4 ) == 0 ) {
      nits = nits / 10;
    }
    if ( nits < 1 ) {
      nits = 1;
    }
    delete [] w;
    delete [] x;
    delete [] y;
    delete [] z;
  }

  if(rank == 0){
  	cout << "\n";
  	cout << "FFT_SERIAL:\n";
  	cout << "  Normal end of execution.\n";
  	cout << "\n";
  	timestamp ( );
  }

  MPI_Finalize();
  
  return 0; 
}

void cffti ( int n, double w[] ){
  double arg;
  double aw;
  int i;
  int n2;
  const double pi = 3.141592653589793;

  n2 = n / 2;
  aw = 2.0 * pi / ( ( double ) n );

  for ( i = 0; i < n2; i++ )
  {
    arg = aw * ( ( double ) i );
    w[i*2+0] = cos ( arg );
    w[i*2+1] = sin ( arg );
  }
  return;
}

double cpu_time ( void ){
  double value;
  value = ( double ) clock ( ) / ( double ) CLOCKS_PER_SEC;
  return value;
}

double ggl ( double *seed ){
  double d2 = 0.2147483647e10;
  double t;
  double value;
  t = *seed;
  t = fmod ( 16807.0 * t, d2 );
  *seed = t;
  value = ( t - 1.0 ) / ( d2 - 1.0 );
  return value;
}

void ccopy ( int n, double x[], double y[] ){
  int i;
  for( i = 0; i < n; i++ ){
    y[i*2+0] = x[i*2+0];
    y[i*2+1] = x[i*2+1];
   }
  return;
}

void timestamp ( ){
# define TIME_SIZE 40
  static char time_buffer[TIME_SIZE];
  const struct tm *tm;
  time_t now;
  now = time ( NULL );
  tm = localtime ( &now );
  strftime ( time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm );
  cout << time_buffer << "\n";
  return;
# undef TIME_SIZE
}
