#include <vector>
#include <string>

template <class T>
class RaiderSTREAM
{
  public:
    virtual ~RaiderSTREAM(){}

    /* sequential kernels */
    virtual void seq_copy() = 0;
    virtual void seq_scale() = 0;
    virtual void seq_sum() = 0;
    virtual void seq_triad() = 0;

    /* gather kernels */
    virtual void gather_copy() = 0;
    virtual void gather_scale() = 0;
    virtual void gather_sum() = 0;
    virtual void gather_triad() = 0;

    /* scatter kernels */
    virtual void scatter_copy() = 0;
    virtual void scatter_scale() = 0;
    virtual void scatter_sum() = 0;
    virtual void scatter_triad() = 0;

    /* scatter-gather kernels */
    virtual void sg_copy() = 0;
    virtual void sg_scale() = 0;
    virtual void sg_sum() = 0;
    virtual void sg_triad() = 0;

    /* central kernels */
    virtual void central_copy() = 0;
    virtual void central_scale() = 0;
    virtual void central_sum() = 0;
    virtual void central_triad() = 0;

    /* helper functions */
    virtual double mysecond();
    virtual double checktick();
    virtual void parse_opts(); // FIXME: possibly add in arguments if needed
    virtual void init_random_idx_array(); // FIXME: possibly add in arguments if needed
    virtual void init_read_idx_array(); // FIXME: possibly add in arguments if needed
    virtual void init_stream_array(); // FIXME: possibly add in arguments if needed
};














// /*
// TODO: document mysecond()
// */
// double mysecond() {
//   struct timeval tp;
//   struct timezone tzp;
//   int i;
//   i = gettimeofday(&tp,&tzp);
//   return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
// }

// /*
// TODO: document checktick()
// */
// # define	M	20
// int checktick()
// {
//   int		i, minDelta, Delta;
//   double	t1, t2, timesfound[M];
//   for (i = 0; i < M; i++) {
//     t1 = mysecond();
//     while( ((t2=mysecond()) - t1) < 1.0E-6 );
//     timesfound[i] = t1 = t2;
//   }
//   minDelta = 1000000;
//   for (i = 1; i < M; i++) {
//     Delta = (int)( 1.0E6 * (timesfound[i]-timesfound[i-1]));
//     minDelta = MIN(minDelta, MAX(Delta,0));
// 	}

//   return(minDelta);
// }

// /*
// TODO: document parse_opts()
// */
// void parse_opts(int argc, char **argv, ssize_t *stream_array_size) {
//   int option;
//   while( (option = getopt(argc, argv, "n:t:h")) != -1 ) {
//     switch (option) {
//       case 'n':
//         *stream_array_size = atoi(optarg);
//         break;
//       case 'h':
//         printf("Usage: -n <stream_array_size>\n");
//         exit(2);
//     }
//   }
// }

// /*
// TODO: document init_random_idx_array()
// */
// void init_random_idx_array(ssize_t *array, ssize_t nelems) {
// 	int success;
// 	ssize_t i, idx;
// 	char* flags = (char*) malloc(sizeof(char)*nelems); 	// Array to track used indices

// 	for(i = 0; i < nelems; i++){
// 		flags[i] = 0;
// 	}

// 	for (i = 0; i < nelems; i++) { // Iterate and fill each element of the idx array
// 		success = 0;
// 		while(success == 0){
// 			idx = ((ssize_t) rand()) % nelems;
// 			if(flags[idx] == 0){
// 				array[i] = idx;
// 				flags[idx] = -1;
// 				success = 1;
// 			}
// 		}
// 	}

// 	free(flags);
// }

// /*
// TODO: document init_read_idx_array()
// */
// void init_read_idx_array(ssize_t *array, ssize_t nelems, char *filename) {
//     FILE *file;
//     file = fopen(filename, "r");
//     if (!file) {
//       perror(filename);
//       exit(1);
//     }

//     for (ssize_t i=0; i < nelems; i++) {
//       fscanf(file, "%zd", &array[i]);
//     }

//     fclose(file);
// }

// /*
// TODO: document init_stream_array()
// */
// void init_stream_array(STREAM_TYPE *array, ssize_t array_elements, STREAM_TYPE value) {
//   #pragma omp parallel for
//   for (ssize_t i = 0; i < array_elements; i++) {
//       array[i] = value;
//   }
// }