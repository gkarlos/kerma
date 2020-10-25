# add_executable(SourceLocTest SourceLocTest.cpp)
# target_link_libraries(SourceLocTest gtest gtest_main KermaSourceInfo)
# add_test(NAME SourceInfo.SourceLoc COMMAND SourceLocTest)

# add_executable(SourceRangeTest SourceRangeTest.cpp)
# target_link_libraries(SourceRangeTest gtest gtest_main KermaSourceInfo)
# add_test(NAME SourceInfo.SourceRange COMMAND SourceRangeTest)

add_executable(SourceInfoTest SourceInfoTest.cpp)
target_link_libraries(SourceInfoTest gtest gtest_main KermaSourceInfo)
add_test(NAME SourceInfo.SourceInfo COMMAND SourceInfoTest)