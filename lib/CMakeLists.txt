add_library(MyMain SHARED main.cpp)

if(NOT DEFINED PYTORCH_LIBS_PATH)
  message(FATAL_ERROR "PYTORCH_LIBS_PATH required.")
endif()

target_link_directories(MyMain PUBLIC
  "${PYTORCH_LIBS_PATH}")

target_link_libraries(MyMain
  c10 torch_cpu)

install(TARGETS MyMain DESTINATION lib)
