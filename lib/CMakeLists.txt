add_library(MyMain SHARED main.cpp)


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
