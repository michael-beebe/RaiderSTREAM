#ifndef _RSBASEIMPL_H_
#define _RSBASEIMPL_H_

#include <iostream>
#include <stdlib.h>
// #include <bits/stdc++.h>
#include <sys/time.h>
#include <stdint.h>
#include <sys/types.h>

#ifndef NUM_KERNELS
#define NUM_KERNELS 20 // TODO: possibly find a better place for this
#endif

# ifndef NUM_ARRAYS
# define NUM_ARRAYS 3
# endif

# ifndef MIN
# define MIN(x,y) ((x)<(y)?(x):(y))
# endif
# ifndef MAX
# define MAX(x,y) ((x)>(y)?(x):(y))
# endif

#ifndef abs
#define abs(a) ((a) >= 0 ? (a) : -(a))
#endif

#define	M	20

class RSBaseImpl {
public:
  typedef enum {
    RS_NB     = 0,
    RS_ALL    = 1,

    RS_SEQ_COPY = 2,
    RS_SEQ_SCALE = 3,
    RS_SEQ_ADD = 4,
    RS_SEQ_TRIAD = 5,
    
    RS_GATHER_COPY = 6,
    RS_GATHER_SCALE = 7,
    RS_GATHER_ADD = 8,
    RS_GATHER_TRIAD = 9,
    
    RS_SCATTER_COPY = 10,
    RS_SCATTER_SCALE = 11,
    RS_SCATTER_ADD = 12,
    RS_SCATTER_TRIAD = 13,
    
    RS_SG_COPY = 14,
    RS_SG_SCALE = 15,
    RS_SG_ADD = 16,
    RS_SG_TRIAD = 17,
    
    RS_CENTRAL_COPY = 18,
    RS_CENTRAL_SCALE = 19,
    RS_CENTRAL_ADD = 20,
    RS_CENTRAL_TRIAD = 21
  } RSKernelType;

  // TODO: Default Constructor
  RSBaseImpl(
    // TODO: fill in these args
  ) {}
  
  virtual ~RSBaseImpl() {}
  
  virtual bool allocate_data( // TODO: finishing adding things that need to be allocated
    double *a,
    double *b,
    double *c,
    ssize_t *IDX1,
    ssize_t *IDX2,
    ssize_t *IDX3,
    double *avgtime,
    double *maxtime,
    double *mintime
  ) = 0;
  
  virtual bool free_data( // TODO: finish adding things that need to be freed
    double *a,
    double *b,
    double *c,
    ssize_t *IDX1,
    ssize_t *IDX2,
    ssize_t *IDX3,
    double *avgtime,
    double *maxtime,
    double *mintime
  ) = 0;
  
  /**
    * Execute the benchmarks
    * 
  */
  virtual bool execute(/*TODO: fill in these args*/) = 0;
  
////////////////////////////////////////////////////////////////
//        TODO: DEAL WITH VALIDATION LATER
////////////////////////////////////////////////////////////////
  virtual bool check_errors(
    const char *label,
    double *array, // FIXME: STREAM_TYPE
    double avg_err, // FIXME: STREAM_TYPE
		double exp_val, // FIXME: STREAM_TYPE
		double epsilon,
		int *errors,
		ssize_t stream_array_size
	) = 0;
	
	virtual bool central_check_errors(
	  const char *label,
	  double *array, // FIXME: STREAM_TYPE
	  double avg_err, // FIXME: STREAM_TYPE
		double exp_val, // FIXME: STREAM_TYPE
		double epsilon,
		int *errors,
		ssize_t stream_array_size
	) = 0;
	
	virtual bool standard_errors(
	  double aj, // FIXME: STREAM_TYPE
	  double bj, // FIXME: STREAM_TYPE
	  double cj, // FIXME: STREAM_TYPE
	  ssize_t stream_array_size,
	  double *a, // FIXME: STREAM_TYPE
	  double *b, // FIXME: STREAM_TYPE
	  double *c, // FIXME: STREAM_TYPE
	  double *aSumErr, // FIXME: STREAM_TYPE
	  double *bSumErr, // FIXME: STREAM_TYPE
	  double *cSumErr, // FIXME: STREAM_TYPE
	  double *aAvgErr, // FIXME: STREAM_TYPE
	  double *bAvgErr, // FIXME: STREAM_TYPE
	  double *cAvgErr // FIXME: STREAM_TYPE
	) = 0;
	
	virtual bool central_errors(
	  double aj, // FIXME: STREAM_TYPE
	  double bj, // FIXME: STREAM_TYPE
	  double cj, // FIXME: STREAM_TYPE
	  ssize_t stream_array_size,
	  double *a, // FIXME: STREAM_TYPE
	  double *b, // FIXME: STREAM_TYPE
	  double *c, // FIXME: STREAM_TYPE
	  double *aSumErr, // FIXME: STREAM_TYPE
	  double *bSumErr, // FIXME: STREAM_TYPE
	  double *cSumErr, // FIXME: STREAM_TYPE
	  double *aAvgErr, // FIXME: STREAM_TYPE
	  double *bAvgErr, // FIXME: STREAM_TYPE
	  double *cAvgErr // FIXME: STREAM_TYPE
	) = 0;
	
	virtual bool validate_values(
	  double aj, // FIXME: STREAM_TYPE
	  double bj, // FIXME: STREAM_TYPE
	  double cj, // FIXME: STREAM_TYPE
	  ssize_t stream_array_size,
	  double *a, // FIXME: STREAM_TYPE
	  double *b, // FIXME: STREAM_TYPE
	  double *c // FIXME: STREAM_TYPE
	  // KernelType KType
	) = 0;
	
	virtual bool seq_validation(
	  ssize_t stream_array_size,
	  double scalar, // FIXME: STREAM_TYPE
	  int *is_validated,
	  double *a, // FIXME: STREAM_TYPE
	  double *b, // FIXME: STREAM_TYPE
	  double *c // FIXME: STREAM_TYPE
	) = 0;
	
	virtual bool gather_validation(
	  ssize_t stream_array_size,
	  double scalar, // FIXME: STREAM_TYPE
	  int *is_validated,
	  double *a, // FIXME: STREAM_TYPE
	  double *b, // FIXME: STREAM_TYPE
	  double *c // FIXME: STREAM_TYPE
	) = 0;
	
	virtual bool scatter_validation(
	  ssize_t stream_array_size,
	  double scalar, // FIXME: STREAM_TYPE
	  int *is_validated,
	  double *a, // FIXME: STREAM_TYPE
	  double *b, // FIXME: STREAM_TYPE
	  double *c // FIXME: STREAM_TYPE
	) = 0;
	
	virtual bool sg_validation(
	  ssize_t stream_array_size,
	  double scalar, // FIXME: STREAM_TYPE
	  int *is_validated,
	  double *a, // FIXME: STREAM_TYPE
	  double *b, // FIXME: STREAM_TYPE
	  double *c // FIXME: STREAM_TYPE
	) = 0;
	
	virtual bool central_validation(
	  ssize_t stream_array_size,
	  double scalar, // FIXME: STREAM_TYPE
	  int *is_validated,
	  double *a, // FIXME: STREAM_TYPE
	  double *b, // FIXME: STREAM_TYPE
	  double *c // FIXME: STREAM_TYPE
	) = 0;
	
	virtual bool checkSTREAMresults(int *is_validated) = 0;
	////////////////////////////////////////////////////////////////
  /**
    * Initializes an array with unique random indices.
    * 
    * @param array Pointer to the array to be initialized.
    * @param nelems Number of elements in the array.
  */
  void init_random_idx_array(ssize_t *array, ssize_t nelems) {
  	int success;
  	ssize_t i, idx;
  	char *flags = (char *) malloc(sizeof(char) * nelems);
  	for (i = 0; i < nelems; i++)
  		flags[i] = 0;
  	for (i = 0; i < nelems; i++) {
  		success = 0;
  		while (success == 0) {
  			idx = ((ssize_t)rand()) % nelems;
  			if (flags[idx] == 0) {
  				array[i] = idx;
  				flags[idx] = -1;
  				success = 1;
  			}
  		}
  	}
  	free(flags);
  }
  
  /**
    * Reads indices from a file and initializes an array with them.
    * 
    * @param array Pointer to the array to be initialized.
    * @param nelems Number of elements in the array.
    * @param filename Name of the file containing the indices.
  */
  void init_read_idx_array(ssize_t *array, ssize_t nelems, char *filename) {
  	FILE *file;
  	file = fopen(filename, "r");
  	if (!file) {
  		perror(filename);
  		exit(1);
  	}
  	for (ssize_t i = 0; i < nelems; i++)
  		fscanf(file, "%zd", &array[i]);
  
  	fclose(file);
  }
  
  /**
    * Initializes an array with a given value.
    * 
    * @param array Pointer to the array to be initialized.
    * @param array_elements Number of elements in the array.
    * @param value The value to initialize each element with.
  */
  void init_stream_array(double *array, ssize_t array_elements, double value) {
    for (ssize_t i = 0; i < array_elements; i++)
      array[i] = value;
  }
  
  /**
    * Returns the current time in seconds since the epoch.
    * 
    * @return The current time in seconds as a double.
  */
  double my_second() {
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp,&tzp);
    return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
  }
  
  /**
    * Checks and returns the time resolution of the timer in use.
    * 
    * @return Minimum time difference found after M tries.
  */
  #define M 20
  int checktick() {
    int i, minDelta, Delta;
    double t1, t2, timesfound[M];
    for (i = 0; i < M; i++) {
      t1 = my_second();
      while (((t2 = my_second()) - t1) < 1.0E-6);
      timesfound[i] = t1 = t2;
    }
    minDelta = 1000000;
    for (i = 1; i < M; i++) {
      Delta = (int)(1.0E6 * (timesfound[i] - timesfound[i - 1]));
      minDelta = MIN(minDelta, MAX(Delta, 0));
    }
    return(minDelta);
  }

  /**
    * Returns the RaiderSTREAM implementation name.
    * 
    * @return Implementation name
  */
  std::string get_impl_name() { return Impl; }
  
  /**
    * Returns the time resolution of the timer in use.
    * 
    * @return Kernelk Name
  */
  RSBaseImpl::RSKernelType get_kernel_type() { return KType; }



private:
  std::string Impl;
  RSBaseImpl::RSKernelType KType;
};

#endif // _RSBASEIMPL_H_

// EOF