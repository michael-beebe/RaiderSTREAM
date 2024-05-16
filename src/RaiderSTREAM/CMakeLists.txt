add_subdirectory(Impl)

include_directories(${RS_INCLUDE_PATH})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set (
  RSSrcs
  RS_Main.cpp
  RSOpts.cpp
)

if (ENABLE_OMP)
  add_executable(raiderstream ${RSSrcs} $<TARGET_OBJECTS:RS_OMP_OBJS>)
endif()

if (ENABLE_MPI_OMP)
  add_executable(raiderstream ${RSSrcs} $<TARGET_OBJECTS:RS_MPI_OMP_OBJS>)
endif()

if (ENABLE_OPENSHMEM)
  add_executable(raiderstream ${RSSrcs} $<TARGET_OBJECTS:RS_OPENSHMEM_OBJS>)
endif()

if (ENABLE_CUDA)
  add_executable(raiderstream ${RSSrcs} $<TARGET_OBJECTS:RS_CUDA_OBJS>)
endif()

if (ENABLE_MPI_CUDA)
  add_executable(raiderstream ${RSSrcs} $<TARGET_OBJECTS:RS_MPI_CUDA_OBJS>)
endif()

install(TARGETS raiderstream DESTINATION bin)