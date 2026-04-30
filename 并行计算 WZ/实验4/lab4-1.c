# include<stdio.h>
# include<stdlib.h>
# include<pthread.h>
#include <time.h>
#include<math.h>

typedef struct{
	double a,b,c;
	double x1,x2;
	double delta;
	double sqrt;
	pthread_cond_t cond;
	pthread_mutex_t mutex;
}equasion;

void* compute_delta(void *arg){
	equasion *eq = (equasion*)arg;
	eq->delta = eq->b * eq->b - 4 * eq->a * eq->c;
//	signal to main thread 
	pthread_cond_signal(&eq->cond); 
	return NULL;
}

void* compute_sqrt(void *arg){
	equasion *eq = (equasion*)arg;
//	add lock on eq,
	pthread_mutex_lock(&eq->mutex);
	while(eq->delta==-1){
//		if delta not computed, wait
		pthread_cond_wait(&eq->cond,&eq->mutex);
	}
	eq->sqrt = sqrt(eq->delta);
//	signal to main thread
	pthread_cond_signal(&eq->cond); 
//	release lock on eq, let other threads visit eq
    pthread_mutex_unlock(&eq->mutex);
    return NULL;
}

void *compute_ans(void *arg){
	equasion *eq = (equasion*)arg;
	//	add lock on eq,
	pthread_mutex_lock(&eq->mutex);
	while(eq->sqrt==-1){
//		if sqrt not computed, wait
		pthread_cond_wait(&eq->cond,&eq->mutex);
	}
	eq->x1 = (-eq->b + eq->sqrt) / (2 * eq->a);
	eq->x2 = (-eq->b - eq->sqrt) / (2 * eq->a);
	//	release lock on eq, let other threads visit eq
    pthread_mutex_unlock(&eq->mutex);
    return NULL;
}

int main(int argc, char *argv[]){
	double a = atoi(argv[1]);
	double b = atoi(argv[2]);
	double c = atoi(argv[3]);
	
	equasion eq = {a,b,c,0,0,-1,-1};
	pthread_cond_init(&eq.cond,NULL);
	pthread_mutex_init(&eq.mutex,NULL);
	
	pthread_t *thread1 = (pthread_t *)malloc(sizeof(pthread_t));
	pthread_t *thread2 = (pthread_t *)malloc(sizeof(pthread_t));
	pthread_t *thread3 = (pthread_t *)malloc(sizeof(pthread_t));
	
	clock_t start_time = clock();
	
	pthread_create(&thread1,NULL,compute_delta,&eq);
	pthread_create(&thread2,NULL,compute_sqrt,&eq);
	pthread_create(&thread3,NULL,compute_ans,&eq);
	
	pthread_join(thread1,NULL);
	pthread_join(thread2,NULL);
	pthread_join(thread3,NULL);
	
	clock_t end_time = clock();
	double runtime = ((double)(end_time-start_time))/CLOCKS_PER_SEC; 
	
	if(eq.delta>=0){
		printf("answer: x1 = %.6f,x2 = %.6f, runtime = %.6fs\n",eq.x1,eq.x2,runtime);
	}
	else{
		printf("no real answer,runtime = %.6fs\n",runtime);
	}
	
	pthread_cond_destroy(&eq.cond);
	pthread_mutex_destroy(&eq.mutex);
	//free(thread1);
	//free(thread2);
	//free(thread3);
	
	return 0;
	
}
