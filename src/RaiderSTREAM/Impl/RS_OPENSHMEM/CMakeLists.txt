set(RSSHMEMSrcs
)

if (ENABLE_OPENSHMEM)
  set(RSSHMEMSrcs ${RSSHMEMSrcs} RS_SHMEM.h RS_SHMEM.cpp RS_SHMEM_IMPL.c)
endif()

include_directories(${RS_INCLUDE_PATH})
include_directories(./)

add_library(RS_SHMEM_OBJS OBJECT ${RSSHMEMSrcs})