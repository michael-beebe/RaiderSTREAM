#ifndef STREAM_OPENMP_H
#define STREAM_OPENMP_H

# include <stdio.h>
# include <stdlib.h>
# include <unistd.h>
# include <math.h>
# include <float.h>
# include <limits.h>
# include <sys/time.h>
# include <time.h>
# include <getopt.h>

# define HLINE "---------------------------------------------------------------------------------------\n"

#ifdef NTIMES
#if NTIMES<=1
#   define NTIMES	10
#endif
#endif
#ifndef NTIMES
#   define NTIMES	10
#endif

#ifndef OFFSET
#   define OFFSET	0
#endif

# ifndef MIN
# define MIN(x,y) ((x)<(y)?(x):(y))
# endif
# ifndef MAX
# define MAX(x,y) ((x)>(y)?(x):(y))
# endif

#ifndef STREAM_TYPE
#define STREAM_TYPE double
#endif

/*--------------------------------------------------------------------------------------
- Specifies the total number of benchmark kernels
- This is important as it is used throughout the benchmark code
--------------------------------------------------------------------------------------*/
# ifndef NUM_KERNELS
# define NUM_KERNELS 12
# endif

/*--------------------------------------------------------------------------------------
- Specifies the total number of stream arrays used in the main loop
--------------------------------------------------------------------------------------*/
# ifndef NUM_ARRAYS
# define NUM_ARRAYS 3
# endif

enum Kernels {
	COPY,
	SCALE,
	SUM,
	TRIAD,
	GATHER_COPY,
	GATHER_SCALE,
	GATHER_SUM,
	GATHER_TRIAD,
	SCATTER_COPY,
	SCATTER_SCALE,
	SCATTER_SUM,
	SCATTER_TRIAD
};

static char *kernel_map[NUM_KERNELS] = {
	"COPY",
	"SCALE",
	"SUM",
	"TRIAD",
	"GATHER_COPY",
	"GATHER_SCALE",
	"GATHER_SUM",
	"GATHER_TRIAD",
	"SCATTER_COPY",
	"SCATTER_SCALE",
	"SCATTER_SUM",
	"SCATTER_TRIAD"
};

#endif