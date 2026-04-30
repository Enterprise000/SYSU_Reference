# include<pthread.h>
# include<stdlib.h>
# include<stdio.h>

typedef struct{
    int start;
    int end;
    int inc;
    void *(*functor)(int, void*);
    void *arg;
    int thread_id;
    int num_threads;
}p_args;

void* p_func(void* arg){
	p_args *args = (p_args* )arg;
	for(int i = args->start + args->thread_id; i<args->end; i += args->num_threads){
		args->functor(i, args->arg);
	}
	return NULL;
}

void parallel_for(int start, int end, int inc, void *(*functor)(int, void*), void *arg, int num_threads){
	pthread_t *threads = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
	p_args * args = (p_args *)malloc(num_threads * sizeof(p_args));
	
	for(int i=0; i<num_threads;i++){
	  args[i].start = start;
          args[i].end = end;
          args[i].inc = inc;
          args[i].functor = functor;
          args[i].arg = arg;
          args[i].thread_id = i;
          args[i].num_threads = num_threads;
        
          pthread_create(&threads[i],NULL,p_func,&args[i]);
	}
	
	for(int j=0; j<num_threads; j++){
		pthread_join(threads[j],NULL);
	}
	
	free(threads);
	free(args);
}
