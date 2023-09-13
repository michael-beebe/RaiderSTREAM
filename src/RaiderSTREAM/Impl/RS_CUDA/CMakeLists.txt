set(CTCUDASrcs
)

if (ENABLE_CUDA)
  set(RSCUDASrcs ${RSCUDASrcs} RS_CUDA.cuh RS_CUDA.cu RS_CUDA_IMPL.cu)
endif()

include_directories(${RS_INCLUDE_PATH})
include_directories(./)

add_library(RS_CUDA_OBJS OBJECT ${RSCUDASrcs})
set_property(TARGET RS_CUDA_OBJS PROPERTY CUDA_ARCHITECTURES OFF) # FIXME: check if this needs to be turned on