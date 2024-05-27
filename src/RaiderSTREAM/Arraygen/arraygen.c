# include <stdio.h>
# include <stdlib.h>
# include <unistd.h>
# include <math.h>
# include <float.h>
# include <limits.h>
# include <sys/time.h>
# include <time.h>

#ifndef STREAM_ARRAY_SIZE
#   define STREAM_ARRAY_SIZE	10000000
#endif

static int IDX[STREAM_ARRAY_SIZE];

/*
------------------------------------------------------------
	DEFINITION OF YOUR CUSTOM FUNCTION FOR POPULATING THE IDX
	ARRAY
------------------------------------------------------------
*/
extern void initUserIdxArray(int *array, int nelems);

int main() {
	initUserIdxArray(IDX, STREAM_ARRAY_SIZE);
	for (int i=0; i<STREAM_ARRAY_SIZE; i++) {
		printf("%d\n", IDX[i]);
	}

	return 0;
}

void initRandomIdxArray(int *array, int nelems) {
	int i, success, idx;

	/* Array to track used indices */
	char* flags = (char*) malloc(sizeof(char)*nelems);
	for(i = 0; i < nelems; i++){
		flags[i] = 0;
	}

	/* Iterate and fill each element of the idx array */
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

/*
------------------------------------------------------------
  WRITE YOUR FUNCTION TO POPULATE THE IDX ARRAYS HERE
------------------------------------------------------------
*/
extern void initUserIdxArray(int *array, int nelems) {
	/* TODO: Write your function here */
}
