if (ENABLE_OMP)
  add_subdirectory(RS_OMP)
endif()

if (ENABLE_OMP_TARGET)
  add_subdirectory(RS_OMP_TARGET)
endif()

if (ENABLE_MPI_OMP)
  add_subdirectory(RS_MPI_OMP)
endif()

if (ENABLE_SHMEM_OMP)
  add_subdirectory(RS_SHMEM_OMP)
endif()

if (ENABLE_SHMEM_OMP_TARGET)
  add_subdirectory(RS_SHMEM_OMP_TARGET)
endif()

if (ENABLE_SHMEM_CUDA)
  add_subdirectory(RS_SHMEM_CUDA)
endif()

if (ENABLE_SHMEM_OACC)
  add_subdirectory(RS_SHMEM_OACC)
endif()

if (ENABLE_CUDA)
  add_subdirectory(RS_CUDA)
endif()

if (ENABLE_OACC)
  add_subdirectory(RS_OACC)
endif()
