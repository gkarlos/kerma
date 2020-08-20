#include "kerma/Base/Dim.h"
#include "kerma/Base/Index.h"

#include "gtest/gtest.h"
#include <stdexcept>
#include <system_error>

using namespace kerma;

namespace {

TEST(IndexTest, Init) {
  Index idx0;
  EXPECT_EQ(idx0.x, 0);
  EXPECT_EQ(idx0.y, 0);
  EXPECT_EQ(idx0.z, 0);

  Index idx1(10);
  EXPECT_EQ(idx1.x, 10);
  EXPECT_EQ(idx1.y, 0);
  EXPECT_EQ(idx1.z, 0);

  Index idx2(10,10);
  EXPECT_EQ(idx2.x, 10);
  EXPECT_EQ(idx2.y, 10);
  EXPECT_EQ(idx2.z, 0);

  Index idx3(10,10,10);
  EXPECT_EQ(idx3.x, 10);
  EXPECT_EQ(idx3.y, 10);
  EXPECT_EQ(idx3.z, 10);
}

TEST(IndexTest, operatorEqual) {
  Index idx;
  EXPECT_TRUE(idx == idx);
  EXPECT_TRUE(idx == Index::Zero);
  EXPECT_TRUE(idx == Index(0,0,0));
  EXPECT_TRUE(Index(0) == Index(0,0));
  EXPECT_TRUE(Index(0,0) == Index(0,0,0));
  EXPECT_TRUE(Index(0) == Index(0,0,0));

  EXPECT_FALSE(Index(0,1) == Index(1,0));
}

TEST(IndexTest, operatorNotEqual) {
  Index idx;
  EXPECT_FALSE(idx != idx);
  EXPECT_FALSE(idx != Index::Zero);
  EXPECT_FALSE(idx != Index(0,0,0));
  EXPECT_FALSE(Index(0) != Index(0,0));
  EXPECT_FALSE(Index(0,0) != Index(0,0,0));
  EXPECT_FALSE(Index(0) != Index(0,0,0));

  EXPECT_TRUE(Index(0,1) != Index(1,0));
}

TEST(IndexTest, operatorLess) {
  Index idx;
  EXPECT_FALSE(idx < idx);

  Index idx1(1);
  Index idx2(2);
  EXPECT_TRUE(idx1 < idx2);

  Index idx3(0,1);
  Index idx4(1);
  EXPECT_FALSE(idx3 < idx4);

  Index idx5(0,1);
  Index idx6(1,0);
  EXPECT_TRUE(idx5 < idx6);

  Index idx7(0,0,1);
  Index idx8(0,0,2);
  Index idx9(0,1,0);
  Index idx10(1,0,0);

  EXPECT_TRUE(idx7 < idx8);
  EXPECT_TRUE(idx8 < idx9);
  EXPECT_TRUE(idx9 < idx10);
}

TEST(IndexTest, operatorLessEqual) {
  Index idx;
  EXPECT_TRUE(idx <= idx);

  Index idx1(1);
  Index idx2(2);
  EXPECT_TRUE(idx1 <= idx2);

  Index idx3(0,1);
  Index idx4(1);
  EXPECT_TRUE(idx3 <= idx4);

  Index idx5(0,1);
  Index idx6(1,0);
  EXPECT_TRUE(idx5 <= idx6);

  Index idx7(0,0,1);
  Index idx8(0,0,2);
  Index idx9(0,1,0);
  Index idx10(1,0,0);

  EXPECT_TRUE(idx7 <= idx8);
  EXPECT_TRUE(idx8 <= idx9);
  EXPECT_TRUE(idx9 <= idx10);

  EXPECT_FALSE(Index(3,1,1) <= Index(2,2,2));
}

TEST(IndexTest, operatorGreater) {
  Index idx;
  EXPECT_FALSE(idx > idx);

  Index idx1(1,1);
  Index idx2(1,2);
  EXPECT_TRUE(idx2 > idx1);
  Index idx3(2,1);
  EXPECT_FALSE(idx2 > idx3);

  EXPECT_TRUE(Dim(1,2,1) > Dim(1,2,0));
  EXPECT_FALSE(Dim(1,1,2) > Dim(1,2));
  EXPECT_FALSE(Dim(0,1,2) > Dim(1,2));
  EXPECT_FALSE(Dim(0,1,2) > Dim(1,1));
}

TEST(IndexTest, operatorGreaterEqual) {
  Index idx;
  EXPECT_TRUE(idx >= idx);

  Index idx1(1,1);
  Index idx2(1,2);
  EXPECT_TRUE(idx2 >= idx1);
  Index idx3(2,1);
  EXPECT_FALSE(idx2 >= idx3);

  EXPECT_TRUE(Dim(1,2,1) >= Dim(1,2,0));
  EXPECT_TRUE(Dim(1,1,2) >= Dim(1,2));

  EXPECT_FALSE(Dim(0,1,2) >= Dim(1,2));
  EXPECT_FALSE(Dim(0,1,2) >= Dim(1,1));
  EXPECT_TRUE(Dim(1,15,20) >= Dim(1,15,20));
}

TEST(IndexTest, operatorPlusPlus) {
  Index idx;
  idx++;
  EXPECT_EQ(idx.x, 1);

  Index idx2(1,1,1);
  EXPECT_EQ(idx++.x, 2);
  EXPECT_EQ((++idx).x, 3);

  Index idx3(1,1,1);
  Index idx4(1,1,1);
  idx3++;
  ++idx4;
  EXPECT_TRUE(idx3 == idx4);
  EXPECT_TRUE(idx3.x == idx4.x);
}

TEST(IndexTest, operatorMinusMinus) {
  //TODO
}

TEST(IndexTest, operatorPlus) {
  Index idx;
  EXPECT_TRUE(idx == (idx + idx));
  Index idx2(1,2,3);
  EXPECT_TRUE((idx2 + idx2) == Index(2,4,6));
}

TEST(IndexTest, operatorMinus) {
  Index idx;
  EXPECT_TRUE(idx == (idx - idx));
  Index idx2(1,2,3);
  EXPECT_TRUE((idx2 - idx2) == Index(0,0,0));
  Index idx3(3,3,3);
  EXPECT_TRUE(idx3 - Index(1,1,1) == Index(2,2,2));
  EXPECT_TRUE(idx3 - Index(1) == Index(3,3,2));
  EXPECT_TRUE((idx3 - Index(1,1,1))-- == Index(2,2,1));
}

TEST(IndexTest, operatorPlusEqual) {
  Index idx;
  idx += idx;
  EXPECT_TRUE(idx == idx);

  idx += Index(1,1,1);
  EXPECT_TRUE(idx == Index(1,1,1));

  idx += Index(1,2,3);
  EXPECT_TRUE(idx == Index(2,3,4));
}

TEST(IndexTest, operatorMinusEqual) {
  Index idx;
  idx -= idx;
  EXPECT_TRUE(idx == idx);

  idx += Index(1,2,3);
  idx -= Index(1,1,1);
  EXPECT_TRUE(idx == Index(0,1,2));
  EXPECT_TRUE(idx == Index(1,2));
}

TEST(IndexTest, inc) {
  Index idx;
  idx.inc(1,1,1);

  EXPECT_TRUE(idx == Index(1,1,1));
  EXPECT_TRUE(Index(0,0,0).inc(1) == (Index(0,0,0)++));
  EXPECT_TRUE(Index().inc(1).inc(1,2).inc(2,2,2) == Index(2,3,5));
}

TEST(IndexTest, dec) {
  Index idx(1,1,1);
  EXPECT_TRUE(idx.dec(1) == Index(1,1,0));
  EXPECT_TRUE(Index(1,1,1).dec(1,0,0).dec(0,1,0).dec(1) == Index::Zero);
}

TEST(IndexTest, getLinear) {
  EXPECT_EQ(Index(9,9,9).getLinear(Dim(10,10,10)), 999);
  EXPECT_THROW(Index(10,10,10).getLinear(Dim(10,10,10)), std::out_of_range);
}

TEST(IndexTest, linearize_STATIC) {
  auto &linearize = Index::linearize;
  EXPECT_EQ(linearize(Index(1), Dim(2,2,2)), 1);
  EXPECT_EQ(linearize(Index(9,9,9), Dim(10,10,10)), 999);
  EXPECT_THROW(linearize(Index(10,10,10), Dim(10,10,10)), std::out_of_range);
}

TEST(IndexTest, delinearize_STATIC) {
  EXPECT_TRUE(Index::delinearize(999, Dim(10,10,10)) == Index(9,9,9));
  EXPECT_TRUE(Index::delinearize(0, Dim(123,1232,3321)) == Index(0));
  EXPECT_TRUE(Index::delinearize(1, Dim(10,10,1)) == Index(1));
  EXPECT_TRUE(Index::delinearize(1, Dim(10,10,1)) == Index(0,1));
  EXPECT_TRUE(Index::delinearize(1, Dim(10,10,1)) == Index(0,0,1));

  EXPECT_TRUE(Index::delinearize(1, Dim(1,10)) == Index(1,0));

  EXPECT_THROW(Index::delinearize(1000, Dim(10,10)), std::out_of_range);
}

} // namespace end