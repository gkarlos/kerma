add_executable(DimTest DimTest.cpp)
target_link_libraries(DimTest gtest gtest_main KermaBase LLVMSupport)
add_test(NAME Base.Dim COMMAND DimTest)

add_executable(IndexTest IndexTest.cpp)
target_link_libraries(IndexTest gtest gtest_main KermaBase LLVMSupport)
add_test(NAME Base.Index COMMAND IndexTest)