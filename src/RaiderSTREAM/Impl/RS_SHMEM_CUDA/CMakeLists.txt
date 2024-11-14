set(RSSHMEMCUDASrcs
)

if (ENABLE_SHMEM_CUDA)
  set(RSSHMEMCUDASrcs ${RSSHMEMCUDASrcs} RS_SHMEM_CUDA.cuh RS_SHMEM_CUDA.cu RS_SHMEM_CUDA_IMPL.cu)
endif()

include_directories(${RS_INCLUDE_PATH})
include_directories(./)

add_library(RS_SHMEM_CUDA_OBJS OBJECT ${RSSHMEMCUDASrcs})
set_property(TARGET RS_SHMEM_CUDA_OBJS PROPERTY CUDA_ARCHITECTURES OFF) # FIXME: check if this needs to be turned on
