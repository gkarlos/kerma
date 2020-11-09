add_executable(DimTest DimTest.cpp)
target_link_libraries(DimTest PRIVATE gtest gtest_main Kerma)
add_test(NAME Base.Dim COMMAND DimTest)

# add_executable(IndexTest IndexTest.cpp)
# target_link_libraries(IndexTest PRIVATE gtest gtest_main Kerma)
# add_test(NAME Base.Index COMMAND IndexTest)

# add_executable(MemoryTest MemoryTest.cpp)
# target_link_libraries(MemoryTest gtest gtest_main Kerma)
# add_test(NAME Base.Memory COMMAND MemoryTest)

# add_executable(MemoryAccessTest MemoryTest.cpp)
# target_link_libraries(MemoryAccessTest gtest gtest_main Kerma)
# add_test(NAME Base.MemoryAccess COMMAND MemoryAccessTest)