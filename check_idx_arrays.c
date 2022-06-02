#include<stdio.h>
#include<stdlib.h>
#include<time.h>

#ifndef STREAM_ARRAY_SIZE
#  define STREAM_ARRAY_SIZE	10000000
#endif

static int a_idx[STREAM_ARRAY_SIZE];
static int b_idx[STREAM_ARRAY_SIZE];
static int c_idx[STREAM_ARRAY_SIZE];

void init_idx_array(int *array, int nelems) {
	int i, success, idx;

	// Array to track used indices
	char* flags = (char*) malloc(sizeof(char)*nelems);
	for(i = 0; i < nelems; i++){
		flags[i] = 0;
	}

	// Iterate and fill each element of the idx array
	for (i = 0; i < nelems; i++) {
		success = 0;
		while(success == 0){
			idx = ((int) rand()) % nelems;
			if(flags[idx] == 0){
				array[i] = idx;
				flags[idx] = -1;
				success = 1;
			}
		}
	}
	free(flags);
}

void printRepeating(int arr[], int size) {
    int i, j;
    printf(" Repeating elements are ");
    for(i = 0; i < size-1; i++)
        for(j = i+1; j < size; j++)
            if(arr[i] == arr[j])
                printf(" %d ", arr[i]);
}	

int main() {
    srand(time(0));
    init_idx_array(a_idx, STREAM_ARRAY_SIZE);
    init_idx_array(b_idx, STREAM_ARRAY_SIZE);
    init_idx_array(c_idx, STREAM_ARRAY_SIZE);

    printRepeating(a_idx, STREAM_ARRAY_SIZE);
    printRepeating(b_idx, STREAM_ARRAY_SIZE);
    printRepeating(c_idx, STREAM_ARRAY_SIZE);


    getchar();
    return 0;
}
