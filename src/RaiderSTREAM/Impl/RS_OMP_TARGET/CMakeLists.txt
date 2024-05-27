set(RSOMPTARGETSrcs
)

if (ENABLE_OMP_TARGET)
  set(RSOMPTARGETSrcs ${RSOMPTARGETSrcs} RS_OMP_TARGET.h RS_OMP_TARGET.cpp RS_OMP_TARGET_IMPL.c)
endif()

include_directories(${RS_INCLUDE_PATH})
include_directories(./)

add_library(RS_OMP_TARGET_OBJS OBJECT ${RSOMPTARGETSrcs})
