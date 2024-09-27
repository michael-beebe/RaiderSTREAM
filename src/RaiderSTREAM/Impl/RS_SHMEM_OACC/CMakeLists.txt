set(RSSHMEMOACCSrcs
)

if (ENABLE_SHMEM_OACC)
  set(RSSHMEMOACCSrcs ${RSSHMEMOACCSrcs} RS_SHMEM_OACC.h RS_SHMEM_OACC.cpp RS_SHMEM_OACC_IMPL.c)
endif()

include_directories(${RS_INCLUDE_PATH})
include_directories(./)

add_library(RS_SHMEM_OACC_OBJS OBJECT ${RSSHMEMOACCSrcs})
