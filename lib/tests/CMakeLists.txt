function(add_tdnat_test NAME)
  add_executable(${NAME} ${NAME}.cpp)

  target_link_libraries(${NAME} tdnat)

  set_target_properties(${NAME} PROPERTIES
    CXX_STANDARD 14)
endfunction()

add_tdnat_test(link_test)
