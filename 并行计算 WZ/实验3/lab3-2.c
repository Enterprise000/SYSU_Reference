#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

typedef struct{
	int *arr;
	int *part_sum;
	int start;
	int end;
}thread_arr;

void* sum(void* arg) {
    thread_arr *data = (thread_arr*)arg;
    int sum = 0;
    for (int i = data->start; i < data->end; i++) {
        sum += data->arr[i];
    }
    *(data->part_sum) = sum;
}

int main(int argc, char *argv[]){
	
	int size = atoi(argv[1]);
    int threads = atoi(argv[2]);
    
    //生成数组 
    int *arr = (int*)malloc(size * sizeof(int));
    int sum_verify = 0;
    for(int i=0; i<size; i++){
    	arr[i] = rand() % 1000;
    	sum_verify = sum_verify + arr[i];
	}
	
	//创建线程 
    clock_t start_time = clock();
	pthread_t *thread = (pthread_t *)malloc(threads * sizeof(pthread_t));
    thread_arr* array = (thread_arr *)malloc(threads * sizeof(thread_arr));
    int part_sums[16];
    int size_thread = size / threads;
    for (int i = 0; i < threads; i++) {
        array[i].arr = arr;
        array[i].part_sum = &part_sums[i];
        array[i].start = i * size_thread;
        array[i].end = (i == threads - 1) ? size : (i + 1) * size_thread;
        pthread_create(&thread[i], NULL, sum, (void*)&array[i]);
    }
    
    //合并线程并求和 
    for (int i = 0; i < threads; i++) {
        pthread_join(thread[i], NULL);
    }
    int arr_sum = 0;
    for (int i = 0; i < threads; i++) {
        arr_sum += part_sums[i];
    }
    clock_t end_time = clock();

//  验证    
    if(arr_sum == sum_verify){
    	printf("Calculation Correct!");
	}else{
		printf("Wrong Answer!");
	}
	
    //计算运行时间并输出 
    double runtime = ((double)(end_time-start_time))/CLOCKS_PER_SEC;
    printf("Array Size: %d, Threads: %d, Answer: %d, Runtime: %f seconds\n",size,threads,arr_sum,runtime);
    
    free(arr);
	free(array);
	free(thread);
	return 0;
}
