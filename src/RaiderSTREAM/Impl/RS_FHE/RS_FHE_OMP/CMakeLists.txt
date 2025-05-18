# RS_FHE/RS_FHE_OMP/CMakeLists.txt
cmake_minimum_required(VERSION 3.5)

# Pull in OpenMP
find_package(OpenMP REQUIRED)

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
)

# tell it to compile with OpenMP
target_compile_definitions(RS_FHE_OMP_OBJS PUBLIC _ENABLE_FHE_OMP_)
target_link_libraries(RS_FHE_OMP_OBJS
  PUBLIC
    RS_FHE
    OpenMP::OpenMP_CXX
)