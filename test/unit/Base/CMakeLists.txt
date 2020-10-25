add_executable(DimTest DimTest.cpp)
target_link_libraries(DimTest gtest gtest_main KermaBase)
add_test(NAME Base.Dim COMMAND DimTest)

add_executable(IndexTest IndexTest.cpp)
target_link_libraries(IndexTest gtest gtest_main KermaBase)
add_test(NAME Base.Index COMMAND IndexTest)

add_executable(MemoryTest MemoryTest.cpp)
target_link_libraries(MemoryTest gtest gtest_main KermaBase)
add_test(NAME Base.Memory COMMAND MemoryTest)