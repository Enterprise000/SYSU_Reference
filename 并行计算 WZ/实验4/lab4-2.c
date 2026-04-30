#include<stdio.h>
#include<stdlib.h>
#include<pthread.h>
#include<time.h>


typedef struct{
	int points;
	int points_in;
}Data;

void* es_pai(void *arg){
	srand(time(NULL));
	Data *data = (Data *)arg;
	int points_in = 0;
	
	for(int i=0; i<data->points;i++){
		double x = (double)rand() / RAND_MAX;
		double y = (double)rand() / RAND_MAX;
		if(x*x + y*y <=1){
			points_in++;
		}
	}
	
	data->points_in = points_in;
}

int main(int argc, char *argv[]){
	int n = atoi(argv[1]);
	int num_threads = atoi(argv[2]);
	
	int point_thread = n / num_threads;
	
	pthread_t *threads = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
	Data *data = (Data*)malloc(num_threads * sizeof(Data));
	
	clock_t start_time = clock();
	
	for(int i=0; i<num_threads; i++){
		data[i].points = point_thread;
		pthread_create(&threads[i],NULL,es_pai,&data[i]);
	}
	
	double point_in_all = 0;
	for(int j=0; j<num_threads;j++){
		pthread_join(threads[j],NULL);
		point_in_all +=data[j].points_in;
	}
	
	double pai = 4 * point_in_all / n;
	
	clock_t end_time = clock();
	double runtime = ((double)(end_time-start_time))/CLOCKS_PER_SEC; 
	
	printf("total points : %d, points in circle: %.6f, estimated pai : %.6f, runtime: %.6f",n,point_in_all,pai,runtime);
	
	free(threads);
	free(data);
	
	return 0;
	
}
