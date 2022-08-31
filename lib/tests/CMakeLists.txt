find_package(GTest REQUIRED)

function(add_tdnat_test NAME)
  add_executable(${NAME} ${NAME}.cpp)
  add_test(NAME ${NAME} COMMAND ${NAME})

  target_link_libraries(${NAME}
    tdnat GTest::gtest GTest::gtest_main)

  set_target_properties(${NAME} PROPERTIES
    CXX_STANDARD 14
    COMPILER_FLAGS "-g -O1")
endfunction()

add_tdnat_test(link_test)
add_tdnat_test(jit_identity_test)
