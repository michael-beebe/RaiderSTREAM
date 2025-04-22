cmake_minimum_required(VERSION 3.5)
project(RS_FHE LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

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
    set(CMAKE_EXE_LINKER_FLAGS "${OpenFHE_EXE_LINKER_FLAGS} -static")
    target_link_libraries(RS_FHE ${OpenFHE_STATIC_LIBRARIES})
else()
    add_library(RS_FHE SHARED ${RS_FHE_SRCS} ${RS_FHE_HDRS})
    set(CMAKE_EXE_LINKER_FLAGS ${OpenFHE_EXE_LINKER_FLAGS})
    target_link_libraries(RS_FHE ${OpenFHE_SHARED_LIBRARIES})
endif()

message(STATUS "RS_FHE library built successfully.")
