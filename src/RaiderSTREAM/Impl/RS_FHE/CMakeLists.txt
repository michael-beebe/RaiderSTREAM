cmake_minimum_required(VERSION 3.5)
project(RS_FHE LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

# choose one of CKKS, BFV or BGV at configure time
set(SCHEME "CKKS" CACHE STRING "Select FHE scheme: CKKS, BFV or BGV")
set_property(CACHE SCHEME PROPERTY STRINGS CKKS BFV BGV)

# Option to build static libraries if needed.
option(BUILD_STATIC "Set to ON to build static libraries" OFF)

# Find the OpenFHE package.
find_package(OpenFHE CONFIG REQUIRED)
if (OpenFHE_FOUND)
    message(STATUS "Found OpenFHE")
    message(STATUS "  OpenFHE Version: ${BASE_OPENFHE_VERSION}")
    message(STATUS "  OpenFHE include files: ${OpenFHE_INCLUDE}")
    message(STATUS "  OpenFHE lib files: ${OpenFHE_LIBDIR}")
else()
    message(FATAL_ERROR "OpenFHE package not found!")
endif()

# Add after the find_package(OpenFHE CONFIG REQUIRED) block
message(STATUS "OpenFHE_STATIC_LIBRARIES: ${OpenFHE_STATIC_LIBRARIES}")
message(STATUS "OpenFHE_SHARED_LIBRARIES: ${OpenFHE_SHARED_LIBRARIES}")

get_filename_component(OpenFHE_ROOT ${OpenFHE_INCLUDE} DIRECTORY)

# Add the OpenFHE lib directory to the linker search path
link_directories(${OpenFHE_LIBDIR})

# List the source and header files for RS_FHE module.
set(RS_FHE_SRCS
    RS_FHE.cpp
)

set(RS_FHE_HDRS
    RS_FHE.h
    RS_FHE_Config.h
)

# Create a library (static or shared) for the FHE module.
if(BUILD_STATIC)
    add_library(RS_FHE STATIC ${RS_FHE_SRCS} ${RS_FHE_HDRS})
    target_link_libraries(RS_FHE PUBLIC ${OpenFHE_STATIC_LIBRARIES})
else()
    add_library(RS_FHE SHARED ${RS_FHE_SRCS} ${RS_FHE_HDRS})
    target_link_libraries(RS_FHE PUBLIC ${OpenFHE_SHARED_LIBRARIES})
endif()

# ——— Expose include dirs transitively ———
target_include_directories(RS_FHE
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenFHE_INCLUDE}
    ${OpenFHE_INCLUDE}/pke
    ${OpenFHE_INCLUDE}/core
    ${OpenFHE_INCLUDE}/binfhe
    ${OpenFHE_INCLUDE}/third-party/include
)

# Add this to propagate variables to RS_FHE_OMP
set(OpenFHE_LIBDIR ${OpenFHE_LIBDIR} PARENT_SCOPE)
set(OpenFHE_INCLUDE ${OpenFHE_INCLUDE} PARENT_SCOPE)
set(OpenFHE_SHARED_LIBRARIES ${OpenFHE_SHARED_LIBRARIES} PARENT_SCOPE)

if (ENABLE_FHE_OMP)
  add_subdirectory(RS_FHE_OMP)
endif()

if (ENABLE_FHE_MPI)
  add_subdirectory(RS_FHE_MPI)
endif()

message(STATUS "RS_FHE library built successfully.")