add_executable(DimTest DimTest.cpp)
# target_link_libraries(Test)
target_link_libraries(DimTest gtest gtest_main KermaBase)
add_test(NAME Base.Dim COMMAND DimTest)