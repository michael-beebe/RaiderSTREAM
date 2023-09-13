set(CTCUDASrcs
)

if (ENABLE_MPI_CUDA)
  set(RSMPICUDASrcs ${RSMPICUDASrcs} RS_CUDA.cuh RS_CUDA.cu RS_CUDA_IMPL.cu)
endif()

include_directories(${RS_INCLUDE_PATH})
include_directories(./)

add_library(RS_MPI_CUDA_OBJS OBJECT ${RSMPICUDASrcs})
set_property(TARGET RS_MPI_CUDA_OBJS PROPERTY CUDA_ARCHITECTURES OFF) # FIXME: check if this needs to be turned on