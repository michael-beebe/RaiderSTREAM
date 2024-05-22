set(RSMPISrcs
)

if (ENABLE_MPI_OMP)
  set(RSMPIOMPSrcs ${RSMPIOMPSrcs} RS_MPI_OMP.h RS_MPI_OMP.cpp RS_MPI_OMP_IMPL.c)
endif()

include_directories(${RS_INCLUDE_PATH})
include_directories(./)

add_library(RS_MPI_OMP_OBJS OBJECT ${RSMPIOMPSrcs})