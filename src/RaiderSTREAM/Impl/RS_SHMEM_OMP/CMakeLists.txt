set(RSSHMEMSrcs
)

if (ENABLE_SHMEM_OMP)
  set(RSSHMEMSrcs ${RSSHMEMSrcs} RS_SHMEM_OMP.h RS_SHMEM_OMP.cpp RS_SHMEM_OMP_IMPL.c)
endif()

include_directories(${RS_INCLUDE_PATH})
include_directories(./)

add_library(RS_SHMEM_OBJS OBJECT ${RSSHMEMSrcs})