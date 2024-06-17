set(RSOMPSrcs
)

if (ENABLE_OACC)
  set(RSOACCSrcs ${RSOACCSrcs} RS_OACC.h RS_OACC.cpp RS_OACC_IMPL.c)
endif()

include_directories(${RS_INCLUDE_PATH})
include_directories(./)

add_library(RS_OACC_OBJS OBJECT ${RSOACCSrcs})
