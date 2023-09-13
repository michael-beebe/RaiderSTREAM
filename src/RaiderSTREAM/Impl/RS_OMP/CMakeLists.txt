set(RSOMPSrcs
)

if (ENABLE_OMP)
  set(RSOMPSrcs ${RSOMPSrcs} RS_OMP.h RS_OMP.cpp RS_OMP_IMPL.c)
endif()

include_directories(${RS_INCLUDE_PATH})
include_directories(./)

add_library(RS_OMP_OBJS OBJECT ${RSOMPSrcs})