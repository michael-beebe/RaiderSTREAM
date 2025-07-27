cmake_minimum_required(VERSION 3.5)

message(STATUS "=====================================================")
message(STATUS "STARTING RS_FHE_MPI CMAKE CONFIGURATION")
message(STATUS "=====================================================")

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenFHE CONFIG REQUIRED)

set(RS_FHE_MPI_SRCS
  RS_FHE_MPI.cpp
  RS_FHE_MPI_IMPL.cpp
)

add_library(RS_FHE_MPI_OBJS OBJECT ${RS_FHE_MPI_SRCS})

target_compile_features(RS_FHE_MPI_OBJS PUBLIC cxx_std_17)

target_include_directories(RS_FHE_MPI_OBJS PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_SOURCE_DIR}/src/include/RaiderSTREAM
  ${OpenFHE_INCLUDE}
  ${OpenFHE_INCLUDE}/pke
  ${OpenFHE_INCLUDE}/core
  ${OpenFHE_INCLUDE}/binfhe
  ${OpenFHE_INCLUDE}/third-party/include
)

target_compile_definitions(RS_FHE_MPI_OBJS PUBLIC _ENABLE_FHE_MPI_)

add_library(RS_FHE_MPI STATIC $<TARGET_OBJECTS:RS_FHE_MPI_OBJS>)

target_include_directories(RS_FHE_MPI PUBLIC 
  $<TARGET_PROPERTY:RS_FHE_MPI_OBJS,INTERFACE_INCLUDE_DIRECTORIES>
)
target_compile_definitions(RS_FHE_MPI PUBLIC 
  $<TARGET_PROPERTY:RS_FHE_MPI_OBJS,INTERFACE_COMPILE_DEFINITIONS>
)
target_compile_features(RS_FHE_MPI PUBLIC 
  $<TARGET_PROPERTY:RS_FHE_MPI_OBJS,INTERFACE_COMPILE_FEATURES>
)

target_link_libraries(RS_FHE_MPI
  PUBLIC
    RS_FHE
    OpenMP::OpenMP_CXX
    MPI::MPI_CXX
    ${OpenFHE_LIBDIR}/libOPENFHEpke.so
    ${OpenFHE_LIBDIR}/libOPENFHEcore.so
    ${OpenFHE_LIBDIR}/libOPENFHEbinfhe.so
)
