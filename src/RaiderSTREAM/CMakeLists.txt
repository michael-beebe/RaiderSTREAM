add_subdirectory(Impl)
add_subdirectory(Arraygen)

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

if (ENABLE_OMP_TARGET)
  add_executable(raiderstream ${RSSrcs} $<TARGET_OBJECTS:RS_OMP_TARGET_OBJS>)
endif()

if (ENABLE_OACC)
  add_executable(raiderstream ${RSSrcs} $<TARGET_OBJECTS:RS_OACC_OBJS>)
endif()

if (ENABLE_MPI_OMP)
  add_executable(raiderstream ${RSSrcs} $<TARGET_OBJECTS:RS_MPI_OMP_OBJS>)
endif()

if (ENABLE_SHMEM_OMP)
  add_executable(raiderstream ${RSSrcs} $<TARGET_OBJECTS:RS_SHMEM_OBJS>)
endif()

if (ENABLE_CUDA)
  add_executable(raiderstream ${RSSrcs} $<TARGET_OBJECTS:RS_CUDA_OBJS>)
endif()

if (ENABLE_MPI_CUDA)
  add_executable(raiderstream ${RSSrcs} $<TARGET_OBJECTS:RS_MPI_CUDA_OBJS>)
endif()

install(TARGETS raiderstream DESTINATION bin)

