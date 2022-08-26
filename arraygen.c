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

extern void init_random_idx_array(int *array, int nelems);

// ------------------------------------------------------------
//  DEFINITION OF YOUR CUSTOM FUNCTION FOR POPULATING THE IDX
// 	ARRAY
// ------------------------------------------------------------
extern void init_user_idx_array(int *array, int nelems);
// ------------------------------------------------------------

int main() {
    srand(time(0));
    // init_random_idx_array(IDX, STREAM_ARRAY_SIZE);

    // init_user_idx_array(IDX, STREAM_ARRAY_SIZE);

    for (int i=0; i<STREAM_ARRAY_SIZE; i++) {
        printf("%d\n", IDX[i]);
    }

	return 0;
}

void init_random_idx_array(int *array, int nelems) {
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

// ------------------------------------------------------------
//  WRITE YOUR FUNCTION TO POPULATE THE IDX ARRAYS HERE
// ------------------------------------------------------------
extern void init_user_idx_array(int *array, int nelems) {
    // TODO: Write your function here
}
// ------------------------------------------------------------
