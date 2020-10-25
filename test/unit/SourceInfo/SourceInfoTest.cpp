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


TEST(SourceLocTest, Init) {
  SourceLoc loc(0,10);
  ASSERT_EQ(loc.getLine(), 0);
  ASSERT_EQ(loc.getCol(), 10);
}

TEST(SourceLocTest, InitEmpty) {
  SourceLoc loc;
  ASSERT_EQ(loc.getLine(),0);
  ASSERT_EQ(loc.getCol(), 0);
}

TEST(SourceLocTest, CopyConstructor) {
  SourceLoc loc(10,10);
  SourceLoc cpy(loc);
  ASSERT_EQ(loc.getLine(), cpy.getLine());
  ASSERT_EQ(loc.getCol(), cpy.getCol());
}

TEST(SourceLocTest, MoveConstructor) {
  SourceLoc loc([]() -> SourceLoc { return SourceLoc(10,10);}());
  ASSERT_EQ(loc.getLine(), 10);
  ASSERT_EQ(loc.getCol(), 10);
}

TEST(SourceLocTest, Validation) {
  SourceLoc loc;
  ASSERT_TRUE(loc.isValid());
  ASSERT_FALSE(loc.invalidate().isValid());
  SourceLoc loc1(10);
  ASSERT_TRUE(loc1.isValid());
  SourceLoc loc2(10,10);
  ASSERT_TRUE(loc2.isValid());
}

TEST(SourceLocTest, OperatorBool) {
  SourceLoc loc;
  ASSERT_TRUE(loc);
  ASSERT_FALSE(loc.invalidate());
  SourceLoc loc1(10);
  ASSERT_TRUE(loc1);
  SourceLoc loc2(10,10);
  ASSERT_TRUE(loc2);
}

TEST(SourceLocTest, OperatorAssign) {
  SourceLoc loc(10,10);
  SourceLoc loc2 = loc;
  ASSERT_TRUE(&loc != &loc2);
  ASSERT_TRUE(loc == loc2);
}

TEST(SourceLocTest, OperatorEquals) {
  SourceLoc loc1;
  SourceLoc loc2(10,10);
  SourceLoc loc3(10,10);
  ASSERT_TRUE(loc2 == loc2);
  ASSERT_TRUE(loc2 == loc3);
  ASSERT_FALSE(loc1 == loc2);
}

TEST(SourceLocTest, OperatorNotEqual) {
  SourceLoc loc1(10,10);
  SourceLoc loc2(100,100);
  ASSERT_TRUE(loc1 != loc2);
}

TEST(SourceLocTest, OperatorLT) {
  SourceLoc loc1(10,10);
  SourceLoc loc2(100,100);
  ASSERT_TRUE(loc1 < loc2);
  ASSERT_FALSE(loc1 < loc1);
}

TEST(SourceLocTest, OperatorLTE) {
  SourceLoc loc1(10,10);
  SourceLoc loc2(100,100);
  ASSERT_TRUE(loc1 <= loc2);
  ASSERT_TRUE(loc1 <= loc1);
}

TEST(SourceLocTest, OperatorGT) {
  SourceLoc loc1(10,10);
  SourceLoc loc2(100,100);
  ASSERT_FALSE(loc1 > loc2);
  ASSERT_TRUE(loc2 > loc1);
  ASSERT_FALSE(loc1 > loc1);
}

TEST(SourceLocTest, OperatorGTE) {
  SourceLoc loc1(10,10);
  SourceLoc loc2(100,100);
  ASSERT_FALSE(loc1 >= loc2);
  ASSERT_TRUE(loc2 >= loc1);
  ASSERT_TRUE(loc1 >= loc1);
}

TEST(SourceLocTest, Unknown) {
  ASSERT_FALSE(SourceLoc::Unknown.isValid());
  ASSERT_FALSE(SourceLoc::Unknown);
}


TEST(SourceRangeTest, Init) {
  SourceRange range(SourceLoc(10,0),SourceLoc(10,10));
  EXPECT_EQ(range.getStart(), SourceLoc(10,0));
  EXPECT_EQ(range.getEnd(), SourceLoc(10,10));

  SourceRange range2(10,20);
  EXPECT_EQ(range2.getStart(), SourceLoc(10,0));
  EXPECT_EQ(range2.getEnd(), SourceLoc(20,0));
}

TEST(SourceRangeTest, InitEmpty) {
  SourceRange range;
  EXPECT_TRUE(range.getStart() == range.getEnd());
  EXPECT_TRUE(range.getStart() == SourceLoc());
  EXPECT_TRUE(range.getEnd() == SourceLoc());
}

TEST(SourceRangeTest, IsEmpty) {
  SourceRange range;
  EXPECT_TRUE(range.isEmpty());

  SourceRange range2(10,10);
  EXPECT_TRUE(range2.isEmpty());

  SourceRange range3(SourceLoc(10,10),SourceLoc(10,10));
  EXPECT_TRUE(range3.isEmpty());

  SourceRange range4(10,11);
  EXPECT_FALSE(range4.isEmpty());

  SourceRange range5(SourceLoc(10,10),SourceLoc(10,11));
  EXPECT_FALSE(range5.isEmpty());
}

TEST(SourceRangeTest, OperatorEquals)
{
  SourceRange range;
  EXPECT_TRUE(range == range);

  SourceRange range1(10,10);
  EXPECT_TRUE(range1 == range1);
  EXPECT_FALSE(range == range1);

  SourceRange range2(SourceLoc(10),SourceLoc(10));
  EXPECT_TRUE(range1 == range2);

  SourceRange range3(SourceLoc(10,0), SourceLoc(10,0));
  EXPECT_TRUE(range1 == range3);

  SourceRange range4(SourceLoc(10,10),SourceLoc(10,10));
  SourceRange range5(SourceLoc(10,10),SourceLoc(10,10));
  EXPECT_TRUE(range4 == range5);
}