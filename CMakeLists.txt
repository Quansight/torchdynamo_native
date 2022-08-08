cmake_minimum_required(VERSION 3.22)

project(tdnat VERSION 0.1)

if(NOT DEFINED TORCH_DIR)
  message(FATAL_ERROR "TORCH_DIR required.")
endif()

option(ENABLE_TESTS "Compile the tests." OFF)

find_package(LLVM REQUIRED CONFIG)

include_directories(include)
install(DIRECTORY include/tdnat DESTINATION include)

add_subdirectory(lib)
