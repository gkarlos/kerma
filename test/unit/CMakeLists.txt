add_executable(TestSupportNVVM SupportNVVM.cpp)

target_link_libraries(TestSupportNVVM KermaSupport gtest gtest_main)

set_target_properties(TestSupportNVVM PROPERTIES COMPILE_FLAGS "-fno-rtti")

add_test(NAME Support.NNVM COMMAND TestSupportNVVM)