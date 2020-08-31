#include "kerma/SourceInfo/SourceLoc.h"

#include "gtest/gtest.h"

using namespace kerma;

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