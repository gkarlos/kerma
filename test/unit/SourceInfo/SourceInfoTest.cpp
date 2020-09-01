#include "kerma/SourceInfo/SourceInfo.h"
#include "kerma/SourceInfo/SourceRange.h"

#include "gtest/gtest.h"

using namespace kerma;

TEST(SourceInfoTest, InitEmpty) {
  SourceInfo info;
  ASSERT_EQ(info.getDirectory().size(), 0);
  ASSERT_EQ(info.getFilename().size(), 0);
  ASSERT_EQ(info.getPath().size(), 0);
  ASSERT_EQ(info.getText().size(), 0);
  ASSERT_EQ(info.getRange(), SourceRange::Unknown);
}

TEST(SourceInfoTest, Init) {
  SourceInfo info("this/is/my/source/path.cu", SourceRange(10,10), 
                   "A[idx] = B[idx] * C[idx]");

  ASSERT_EQ(info.getPath(), "this/is/my/source/path.cu");
  ASSERT_EQ(info.getDirectory(), "this/is/my/source");
  ASSERT_EQ(info.getFilename(), "path.cu");
  ASSERT_EQ(info.getText(), "A[idx] = B[idx] * C[idx]");
  ASSERT_EQ(info.getRange(), SourceRange(10,10));
}