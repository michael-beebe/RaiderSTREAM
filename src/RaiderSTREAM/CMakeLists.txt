#------------------------------------------------------------------------------
# Fully‐Homomorphic‐Encryption core library
#------------------------------------------------------------------------------

if(ENABLE_FHE OR ENABLE_FHE_OMP)
  message(STATUS ">>> Adding RS_FHE core helper (Impl/RS_FHE)")
  add_subdirectory(Impl/RS_FHE)

  # make its headers available to everyone
  include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/Impl/RS_FHE
  )
endif()

#------------------------------------------------------------------------------

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

if (ENABLE_SHMEM_OMP_TARGET)
  add_executable(raiderstream ${RSSrcs} $<TARGET_OBJECTS:RS_SHMEM_OMP_TARGET_OBJS>)
endif()

if (ENABLE_SHMEM_CUDA)
  add_executable(raiderstream ${RSSrcs} $<TARGET_OBJECTS:RS_SHMEM_CUDA_OBJS>)
endif()

if (ENABLE_SHMEM_OACC)
  add_executable(raiderstream ${RSSrcs} $<TARGET_OBJECTS:RS_SHMEM_OACC_OBJS>)
endif()

if (ENABLE_CUDA)
  add_executable(raiderstream ${RSSrcs} $<TARGET_OBJECTS:RS_CUDA_OBJS>)
endif()

if (ENABLE_FHE_OMP)
  message(STATUS "ENABLING FHE/OpenMP Implementation")
  add_executable(raiderstream ${RSSrcs} $<TARGET_OBJECTS:RS_FHE_OMP_OBJS>)
  target_link_libraries(raiderstream PRIVATE RS_FHE)
endif()

install(TARGETS raiderstream DESTINATION bin)

