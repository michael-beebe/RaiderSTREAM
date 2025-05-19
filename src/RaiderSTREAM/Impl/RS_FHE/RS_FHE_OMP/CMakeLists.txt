# RS_FHE/RS_FHE_OMP/CMakeLists.txt
cmake_minimum_required(VERSION 3.5)

message(STATUS "=====================================================")
message(STATUS "STARTING RS_FHE_OMP CMAKE CONFIGURATION")
message(STATUS "=====================================================")

# Pull in OpenMP
find_package(OpenMP REQUIRED)

find_package(OpenFHE CONFIG REQUIRED)
if (OpenFHE_FOUND)
    message(STATUS "FOUND PACKAGE OpenFHE")
    message(STATUS "OpenFHE Version: ${BASE_OPENFHE_VERSION}")
    message(STATUS "OpenFHE installed as shared libraries: ${OpenFHE_SHARED}")
    message(STATUS "OpenFHE include files location: ${OpenFHE_INCLUDE}")
    message(STATUS "OpenFHE lib files location: ${OpenFHE_LIBDIR}")
    message(STATUS "OpenFHE Native Backend size: ${OpenFHE_NATIVE_SIZE}")
else()
    message(FATAL_ERROR "PACKAGE OpenFHE NOT FOUND")
endif ()

set( CMAKE_CXX_FLAGS ${OpenFHE_CXX_FLAGS} )

link_directories(${OpenFHE_LIBDIR})
if(BUILD_STATIC)
    set( CMAKE_EXE_LINKER_FLAGS "${OpenFHE_EXE_LINKER_FLAGS} -static")
    link_libraries( ${OpenFHE_STATIC_LIBRARIES} )
else()
    set( CMAKE_EXE_LINKER_FLAGS ${OpenFHE_EXE_LINKER_FLAGS} )
    link_libraries( ${OpenFHE_SHARED_LIBRARIES} )
endif()

# Collect our sources
set(RS_FHE_OMP_SRCS
  RS_FHE_OMP.cpp
  RS_FHE_OMP_IMPL.cpp
)

# Build as an OBJECT library
add_library(RS_FHE_OMP_OBJS OBJECT ${RS_FHE_OMP_SRCS})

# We want C++17
target_compile_features(RS_FHE_OMP_OBJS PUBLIC cxx_std_17)

# expose headers to the final exe
target_include_directories(RS_FHE_OMP_OBJS PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_SOURCE_DIR}/src/include/RaiderSTREAM
  ${OpenFHE_INCLUDE}
  ${OpenFHE_INCLUDE}/pke
  ${OpenFHE_INCLUDE}/core
  ${OpenFHE_INCLUDE}/binfhe
  ${OpenFHE_INCLUDE}/third-party/include
)

# tell it to compile with OpenMP
target_compile_definitions(RS_FHE_OMP_OBJS PUBLIC _ENABLE_FHE_OMP_)

# Create a real library that can be linked against by the top-level CMakeLists.txt
add_library(RS_FHE_OMP STATIC $<TARGET_OBJECTS:RS_FHE_OMP_OBJS>)

# Copy properties from object library to static library
target_include_directories(RS_FHE_OMP PUBLIC 
  $<TARGET_PROPERTY:RS_FHE_OMP_OBJS,INTERFACE_INCLUDE_DIRECTORIES>
)
target_compile_definitions(RS_FHE_OMP PUBLIC 
  $<TARGET_PROPERTY:RS_FHE_OMP_OBJS,INTERFACE_COMPILE_DEFINITIONS>
)
target_compile_features(RS_FHE_OMP PUBLIC 
  $<TARGET_PROPERTY:RS_FHE_OMP_OBJS,INTERFACE_COMPILE_FEATURES>
)

# Link the static library against required dependencies 
target_link_libraries(RS_FHE_OMP
  PUBLIC
    RS_FHE
    OpenMP::OpenMP_CXX
    OpenFHEpke
    OpenFHEcore
    OpenFHEbinfhe
)