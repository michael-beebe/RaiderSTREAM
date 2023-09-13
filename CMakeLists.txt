# RaiderSTREAM Top-Level CMake
#
# michael.beebe@ttu.edu
#
# See LICENSE in the top level directory for licensing details
#

cmake_minimum_required(VERSION 3.4.3)
project(RaiderSTREAM)
set(VERSION "0.2.0")


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

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -std=c99 ${RS_CFLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 ${RS_CXXFLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -Wall -g ${RS_CXXFLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -std=c++11 ${RS_CXXFLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${RS_LINKER_FLAGS}")


#------------------------------------------------------------------------
#-- IMPLEMENTATION-SPECIFIC FLAGS
#------------------------------------------------------------------------
if (ENABLE_OMP)
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_ENABLE_OMP_")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_ENABLE_OMP_")
  message(STATUS "ENABLING OpenMP Implementation")
else()
  message(STATUS "DISABLING OpenMP Implementation")
endif()

if (ENABLE_MPI_OMP)
     set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_ENABLE_MPI_OMP_")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_ENABLE_MPI_OMP_")
  message(STATUS "ENABLING MPI/OpenMP Implementation")
  else()
    message(STATUS "DISABLING MPI/OpenMP Implementation")
endif()

if (ENABLE_OPENSHMEM)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_ENABLE_OPENSHMEM_")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_ENABLE_OPENSHMEM_")
  message(STATUS "ENABLING OpenSHMEM Implementation")
else()
  message(STATUS "DISABLING OpenSHMEM Implementation")
endif()

if (ENABLE_CUDA)
  project(RaiderSTREAM LANGUAGES CUDA CXX C)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -lcudart -lcudadevrt -D_ENABLE_CUDA_")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lcudart -lcudadevrt -D_ENABLE_CUDA_")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lcudart -lcudadevrt -D_ENABLE_CUDA_")   
  set(CMAKE_CUDA_COMPILER "nvcc")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_ENABLE_CUDA_")
  message(STATUS "ENABLING MPI CUDA Implementation")
else()
  message(STATUS "DISABLING MPI CUDA Implementation")
endif()

if (ENABLE_MPI_CUDA)
  project(RaiderSTREAM LANGUAGES CUDA CXX C)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -lcudart -lcudadevrt -D_ENABLE_MPI_CUDA_")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lcudart -lcudadevrt -D_ENABLE_MPI_CUDA_")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lcudart -lcudadevrt -D_ENABLE_MPI_CUDA_")   
  set(CMAKE_CUDA_COMPILER "nvcc")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_ENABLE_MPI_CUDA_")
  message(STATUS "ENABLING MPI CUDA Implementation")
else()
  message(STATUS "DISABLING MPI CUDA Implementation")
endif()


#------------------------------------------------------------------------
# Add package paths
#------------------------------------------------------------------------
if(NOT RS_INCLUDE_PATH)
  set(RS_INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/include")
  message(STATUS "RS_INCLUDE_PATH set to ${RS_INCLUDE_PATH}")
endif()
if(NOT RS_SRC_PATH)
  set(RS_SRC_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src")
  message(STATUS "RS_SRC_PATH set to ${RS_SRC_PATH}")
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
add_subdirectory(include)
add_subdirectory(test)

# EOF