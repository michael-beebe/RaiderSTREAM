set(RSMPISrcs
)

if (ENABLE_MPI_OMP)
  set(RSMPIOMPSrcs ${RSMPIOMPSrcs} RS_MPI.h RS_MPI.cpp RS_MPI_IMPL.c)
endif()

include_directories(${RS_INCLUDE_PATH})
include_directories(./)

add_library(RS_MPI_OMP_OBJS OBJECT ${RSMPIOMPSrcs})