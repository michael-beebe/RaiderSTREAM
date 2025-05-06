cmake_minimum_required(VERSION 3.5)
project(RS_FHE LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

# choose one of CKKS, BFV or BGV at configure time
set(SCHEME "CKKS" CACHE STRING "Select FHE scheme: CKKS, BFV or BGV")
set_property(CACHE SCHEME PROPERTY STRINGS CKKS BFV BGV)

if(SCHEME STREQUAL "CKKS")
  add_definitions(-DCKKS)
elseif(SCHEME STREQUAL "BFV")
  add_definitions(-DBFV)
elseif(SCHEME STREQUAL "BGV")
  add_definitions(-DBG)
else()
  message(FATAL_ERROR "Invalid FHE scheme '${SCHEME}', must be one of CKKS/BFV/BGV")
endif()

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

# Include OpenFHE headers.
include_directories(${OpenFHE_INCLUDE})
include_directories(${OpenFHE_INCLUDE}/third-party/include)
include_directories(${OpenFHE_INCLUDE}/core)
include_directories(${OpenFHE_INCLUDE}/binfhe)
include_directories(${OpenFHE_INCLUDE}/pke)

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
    target_link_libraries(RS_FHE ${OpenFHE_STATIC_LIBRARIES})
else()
    add_library(RS_FHE SHARED ${RS_FHE_SRCS} ${RS_FHE_HDRS})
    target_link_libraries(RS_FHE ${OpenFHE_SHARED_LIBRARIES})
endif()

# Make sure other modules know where to find RS_FHE headers
target_include_directories(RS_FHE PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

message(STATUS "RS_FHE library built successfully.")