#include "kerma/SourceInfo/SourceRange.h"
#include "kerma/SourceInfo/SourceLoc.h"

#include "gtest/gtest.h"

using namespace kerma;

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