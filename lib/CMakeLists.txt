file(GLOB generated_cpp_files
  LIST_DIRECTORIES false CONFIGURE_DEPENDS
  "generated/*.cpp")
list(FILTER generated_cpp_files
  EXCLUDE REGEX ".*Everything\.cpp")

add_library(tdnat SHARED
  function.cpp
  jit_function.cpp
  ops.cpp
  ${generated_cpp_files})

target_link_directories(tdnat PUBLIC
  "${TORCH_DIR}/lib")

target_include_directories(tdnat PUBLIC
  "${TORCH_DIR}/include"
  "${LLVM_INCLUDE_DIRS}")

llvm_map_components_to_libnames(LLVM_LIBS x86asmparser x86codegen orcjit)

target_link_libraries(tdnat
  c10 torch_cpu ${LLVM_LIBS})

set_target_properties(tdnat PROPERTIES
  CXX_STANDARD 14)

install(TARGETS tdnat DESTINATION lib)

if(${ENABLE_TESTS})
  add_subdirectory(tests)
endif()
