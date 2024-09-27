# RaiderSTREAM Top-Level CMake
#
# michael.beebe@ttu.edu
#
# See LICENSE in the top level directory for licensing details
#

cmake_minimum_required(VERSION 3.5.0)
project(RaiderSTREAM)
set(VERSION "0.2.0")

function(use_shmem)
    if (SHMEM_1_5)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_SHMEM_1_5_")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_SHMEM_1_5_")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_SHMEM_1_5_")
    elseif (SHMEM_1_4)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_SHMEM_1_4_")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_SHMEM_1_4_")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_SHMEM_1_4_")
    else()
        message(FATAL_ERROR "Either SHMEM_1_4 or SHMEM_1_5 must be set!")
    endif()
endfunction()

#------------------------------------------------------------------------
#-- EXTERNAL CMAKE SCRIPTS
#------------------------------------------------------------------------
set(CMAKE_MODULE_PATH
  "${CMAKE_SOURCE_DIR}/cmake"
  ${CMAKE_MODULE_PATH}
)
include(UseLATEX OPTIONAL)


#------------------------------------------------------------------------
#-- DERIVE THE RELEASE CODENAME
#------------------------------------------------------------------------
execute_process(COMMAND lsb_release -cs
  OUTPUT_VARIABLE RS_RELEASE_CODENAME
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

#------------------------------------------------------------------------
#-- COMPILER OPTIONS
#------------------------------------------------------------------------
if( RS_CFLAGS )
  message(STATUS "USING MACHINE-SPECIFIC C_FLAGS: ${RS_CFLAGS}")
endif()
if( RS_CXXFLAGS )
  message(STATUS "USING MACHINE-SPECIFIC CXX_FLAGS: ${RS_CXXFLAGS}")
endif()
if( RS_LINKER_FLAGS )
  message(STATUS "USING MACHINE-SPECIFIC LINKER_FLAGS: ${RS_LINKER_FLAGS}")
endif()

set(STREAM_TYPE "double" CACHE STRING "Data type to use for kernels.")
add_definitions(-DSTREAM_TYPE=${STREAM_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -std=c99 ${RS_CFLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -std=c++11 ${RS_CXXFLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -Wall -g ${RS_CXXFLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O0 -std=c++11 ${RS_CXXFLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${RS_LINKER_FLAGS}")

if( DEBUG )
  add_definitions(-D_DEBUG_=1)
endif()

#------------------------------------------------------------------------
#-- IMPLEMENTATION-SPECIFIC FLAGS
#------------------------------------------------------------------------
if (ENABLE_OMP)
  find_package(OpenMP REQUIRED)
  if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -D_ENABLE_OMP_")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -D_ENABLE_OMP_")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    message(STATUS "ENABLING OpenMP Implementation")
  endif()
else()
  message(STATUS "DISABLING OpenMP Implementation")
endif()

if (ENABLE_OMP_TARGET)
  find_package(OpenMP REQUIRED)
  if (OPENMP_FOUND)
    if(NOT DEFINED OFFLOAD_OPTS)
      # TODO: is there a cmake compiler-agnostic way of doing this?
      # summarized from cmake#21784
      set(OFFLOAD_OPTS "")
      set(OFFLOAD_TARGET "nvptx-none")

      if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
        set(OFFLOAD_OPTS "-fopenmp -fopenmp-targets=${OFFLOAD_TARGET}")
      elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
        set(OFFLOAT_OPTS "-fopenmp -foffload=${OFFLOAD_TARGET}")
      elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "NVHPC")
        set(OFFLOAD_OPTS "-mp=gpu")
      elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "PGI")
        set(OFFLOAD_OPTS "-mp=gpu")
      else()
        message(WARNING "Unknown compiler. Please manually set OFFLOAD_OPTS!")
      endif()
    endif()
    message(STATUS "Using ${OFFLOAD_OPTS} to enable offload")

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -D_ENABLE_OMP_TARGET_ ${OFFLOAD_OPTS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -D_ENABLE_OMP_TARGET_ ${OFFLOAD_OPTS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}") #" ${OpenMP_EXE_LINKER_FLAGS}")
    message(STATUS "ENABLING OpenMP + Offload Implementation")
  endif()
else()
  message(STATUS "DISABLING OpenMP + Offload Implementation")
endif()

if (ENABLE_OACC)
  find_package(OpenACC REQUIRED)
  if (1)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenACC_C_FLAGS} -D_ENABLE_OACC_")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenACC_CXX_FLAGS} -D_ENABLE_OACC_")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenACC_EXE_LINKER_FLAGS}")
    message(STATUS "ENABLING OpenACC Implementation")
  endif()
else()
  message(STATUS "DISABLING OpenACC Implementation")
endif()

if (ENABLE_MPI_OMP)
  find_package(MPI REQUIRED)
  find_package(OpenMP REQUIRED)
  if (MPI_FOUND AND OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${MPI_C_FLAGS} ${OpenMP_C_FLAGS} -D_ENABLE_MPI_OMP_")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -D_ENABLE_MPI_OMP_")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    message(STATUS "ENABLING MPI/OpenMP Implementation")
  endif()
else()
  message(STATUS "DISABLING MPI/OpenMP Implementation")
endif()

if (ENABLE_SHMEM_OMP)
  use_shmem()
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_ENABLE_SHMEM_OMP_")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_ENABLE_SHMEM_OMP_")
  message(STATUS "ENABLING OpenSHMEM/OpenMP Implementation")
else()
  message(STATUS "DISABLING OpenSHMEM/OpenMP Implementation")
endif()

if (ENABLE_CUDA)
  enable_language(CUDA)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_ENABLE_CUDA_")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_ENABLE_CUDA_")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O0 -D_ENABLE_CUDA_")
  message(STATUS "ENABLING CUDA Implementation")
else()
  message(STATUS "DISABLING CUDA Implementation")
endif()

# -- hybrids
if (ENABLE_SHMEM_OMP_TARGET)
  if (SHMEM_1_5)
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_SHMEM_1_5_")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_SHMEM_1_5_")
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_SHMEM_1_5_")
  elseif (SHMEM_1_4)
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_SHMEM_1_4_")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_SHMEM_1_4_")
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_SHMEM_1_4_")
  else()
      message(FATAL_ERROR "Either SHMEM_1_4 or SHMEM_1_5 must be set!")
  endif()
  find_package(OpenMP REQUIRED)
  if (OPENMP_FOUND)
    if(NOT DEFINED OFFLOAD_OPTS)
      # TODO: is there a cmake compiler-agnostic way of doing this?
      # summarized from cmake#21784
      set(OFFLOAD_OPTS "")
      set(OFFLOAD_TARGET "nvptx-none")

      if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
        set(OFFLOAD_OPTS "-fopenmp -fopenmp-targets=${OFFLOAD_TARGET}")
      elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
        set(OFFLOAT_OPTS "-fopenmp -foffload=${OFFLOAD_TARGET}")
      elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "NVHPC")
        set(OFFLOAD_OPTS "-mp=gpu")
      elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "PGI")
        set(OFFLOAD_OPTS "-mp=gpu")
      else()
        message(WARNING "Unknown compiler. Please manually set OFFLOAD_OPTS!")
      endif()
    endif()
    message(STATUS "Using ${OFFLOAD_OPTS} to enable offload")

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -D_ENABLE_SHMEM_OMP_TARGET_ ${OFFLOAD_OPTS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -D_ENABLE_SHMEM_OMP_TARGET_ ${OFFLOAD_OPTS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}") #" ${OpenMP_EXE_LINKER_FLAGS}")
    message(STATUS "ENABLING OpenSHMEM/OpenMP Offload Implementation")
  endif()
else()
  message(STATUS "DISABLING OpenSHMEM/OpenMP Offload Implementation")
endif()

if (ENABLE_SHMEM_CUDA)
  if (SHMEM_1_5)
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_SHMEM_1_5_")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_SHMEM_1_5_")
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_SHMEM_1_5_")
  elseif (SHMEM_1_4)
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_SHMEM_1_4_")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_SHMEM_1_4_")
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_SHMEM_1_4_")
  else()
      message(FATAL_ERROR "Either SHMEM_1_4 or SHMEM_1_5 must be set!")
  endif()
  enable_language(CUDA)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_ENABLE_SHMEM_CUDA_")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_ENABLE_SHMEM_CUDA_")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O0 -D_ENABLE_SHMEM_CUDA_")
  message(STATUS "ENABLING OpenSHMEM/CUDA Implementation")
else()
  message(STATUS "DISABLING OpenSHMEM/CUDA Implementation")
endif()

if (ENABLE_SHMEM_OACC)
  if (SHMEM_1_5)
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_SHMEM_1_5_")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_SHMEM_1_5_")
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_SHMEM_1_5_")
  elseif (SHMEM_1_4)
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_SHMEM_1_4_")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_SHMEM_1_4_")
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_SHMEM_1_4_")
  else()
      message(FATAL_ERROR "Either SHMEM_1_4 or SHMEM_1_5 must be set!")
  endif()
  find_package(OpenACC REQUIRED)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenACC_C_FLAGS} -D_ENABLE_SHMEM_OACC_")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenACC_CXX_FLAGS} -D_ENABLE_SHMEM_OACC_")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenACC_EXE_LINKER_FLAGS}")
  message(STATUS "ENABLING OpenSHMEM/OpenACC Implementation")
else()
  message(STATUS "DISABLING OpenSHMEM/OpenACC Implementation")
endif()
# -- end hybrids

#------------------------------------------------------------------------
#-- Add package paths
#------------------------------------------------------------------------
if(NOT RS_SRC_PATH)
  set(RS_SRC_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src")
  message(STATUS "RS_SRC_PATH set to ${RS_SRC_PATH}")
endif()
if(NOT RS_INCLUDE_PATH)
  set(RS_INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src/include")
  message(STATUS "RS_INCLUDE_PATH set to ${RS_INCLUDE_PATH}")
endif()

#------------------------------------------------------------------------
#-- TESTING
#------------------------------------------------------------------------
# Enable testing
enable_testing()


#------------------------------------------------------------------------
#-- PACKAGING
#------------------------------------------------------------------------
#-- builds a standard redhat/centos rpm
if(RAIDERSTREAM_BUILD_RPM)
  set(CPACK_PACKAGE_VERSION ${VERSION})
  set(CPACK_GENERATOR "RPM")
  set(CPACK_PACKAGE_NAME "RaiderSTREAM")
  set(CPACK_PACKAGE_RELEASE 1)
  set(CPACK_PACKAGE_CONTACT "Michael Beebe")
  set(CPACK_PACKAGE_VENDOR "Texas Tech University")
  set(CPACK_RPM_PACKAGE_REQUIRES "devtoolset-7-gcc-c++")
  set(CPACK_PACKAGING_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
  set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CPACK_PACKAGE_RELEASE}.${CMAKE_SYSTEM_PROCESSOR}")
  include(CPack)
endif()

#-- builds a standard debian/ubuntu deb
if(RAIDERSTREAM_BUILD_DEB)
  set(CPACK_PACKAGE_VERSION ${VERSION})
  set(CPACK_GENERATOR "DEB")
  set(CPACK_PACKAGE_NAME "RaiderSTREAM")
  set(CPACK_PACKAGE_RELEASE 1)
  set(CPACK_PACKAGE_CONTACT "Michael Beebe")
  set(CPACK_PACKAGE_VENDOR "Texas Tech University")

  #-- setup package deps
  if( "${CT_RELEASE_CODENAME}" STREQUAL "bionic" )
    message(STATUS "Setting package dependencies for Ubuntu 18.04 (bionic)")
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "")
  elseif( "${CT_RELEASE_CODENAME}" STREQUAL "xenial" )
    message(STATUS "Setting package dependencies for Ubuntu 16.04 (xenial)")
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "")
  else()
    message(STATUS "Unknown debian version: ${CT_RELEASE_CODENAME}")
    message(STATUS "No debian package dependencies are set!!!")
  endif()

  set(CPACK_PACKAGING_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
  set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CPACK_PACKAGE_RELEASE}.${CMAKE_SYSTEM_PROCESSOR}")
  include(CPack)
endif()

#-- builds a tgz
if(RAIDERSTREAM_BUILD_TGZ)
  set(CPACK_PACKAGE_VERSION ${VERSION})
  set(CPACK_GENERATOR "TGZ")
  set(CPACK_PACKAGE_NAME "RaiderSTREAM")
  set(CPACK_PACKAGE_RELEASE 1)
  set(CPACK_PACKAGE_CONTACT "Michael Beebe")
  set(CPACK_PACKAGE_VENDOR "Texas Tech University")
  set(CPACK_PACKAGING_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
  set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CPACK_PACKAGE_RELEASE}.${CMAKE_SYSTEM_PROCESSOR}")
  include(CPack)
endif()

#-- builds an osx dmg
if(RAIDERSTREAM_BUILD_DMG)
  set(CPACK_PACKAGE_VERSION ${VERSION})
  set(CPACK_GENERATOR "DMG")
  set(CPACK_PACKAGE_NAME "RaiderSTREAM")
  set(CPACK_PACKAGE_RELEASE 1)
  set(CPACK_PACKAGE_CONTACT "Michael Beebe")
  set(CPACK_PACKAGE_VENDOR "Texas Tech University")
  set(CPACK_PACKAGING_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
  set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CPACK_PACKAGE_RELEASE}.${CMAKE_SYSTEM_PROCESSOR}")
  include(CPack)
endif()

#------------------------------------------------------------------------
#-- SUBDIRECTORIES
#------------------------------------------------------------------------
# Add subdirectories
add_subdirectory(src)
add_subdirectory(test)

# EOF

