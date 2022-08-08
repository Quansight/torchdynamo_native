cmake_minimum_required(VERSION 3.22)

project(tdnat VERSION 0.1)

include_directories(include)
add_subdirectory(lib)

install(DIRECTORY include/tdnat DESTINATION include)
