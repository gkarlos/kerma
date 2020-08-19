#include "kerma/Base/Dim.h"
#include "kerma/Base/Index.h"

#include "gtest/gtest.h"
#include <stdexcept>
#include <system_error>

using namespace kerma;

namespace {

TEST(DimTest, Init) {
  Dim dim(1,2,3);
  ASSERT_EQ(dim.x, 1);
  ASSERT_EQ(dim.y, 2);
  ASSERT_EQ(dim.z, 3);
}

TEST(DimTest, InitEmpty) {
  Dim dim;
  ASSERT_EQ(dim.x, 1);
  ASSERT_EQ(dim.y, 1);
  ASSERT_EQ(dim.z, 1);
  EXPECT_TRUE(dim);
}

TEST(DimTest, None) {
  ASSERT_EQ(Dim::None.x, 0);
  ASSERT_EQ(Dim::None.y, 0);
  ASSERT_EQ(Dim::None.z, 0);
  EXPECT_FALSE(Dim::None);
}

TEST(DimTest, Size) {
  Dim dim;
  EXPECT_EQ(dim.size(), 1);

  Dim dim2(1,1,1);
  EXPECT_EQ(dim2.size(), 1);

  Dim dim3(10,10,10);
  EXPECT_EQ(dim3.size(), 10 * 10 * 10);
}

TEST(DimTest, is1D) {
  Dim dim1;
  EXPECT_TRUE(dim1.is1D());
  Dim dim2(2);
  EXPECT_TRUE(dim2.is1D());
  Dim dim3(1,2,3);
  EXPECT_FALSE(dim3.is1D());
  EXPECT_FALSE(Dim::None.is1D());
}

TEST(DimTest, is2D) {
  Dim dim1;
  EXPECT_FALSE(dim1.is2D());
  Dim dim2(2);
  EXPECT_FALSE(dim2.is2D());
  Dim dim3(10,10);
  EXPECT_TRUE(dim3.is2D());
  Dim dim4(1,10);
  EXPECT_TRUE(dim4.is2D());
  Dim dim5(1,2,3);
  EXPECT_FALSE(dim5.is2D());
  EXPECT_FALSE(Dim::None.is2D());
}

TEST(DimTest, is3D) {
  Dim dim1;
  EXPECT_FALSE(dim1.is3D());
  Dim dim2(2);
  EXPECT_FALSE(dim2.is3D());
  Dim dim3(1,2);
  EXPECT_FALSE(dim3.is3D());
  Dim dim4(1,1,10);
  EXPECT_TRUE(dim4.is3D());
  Dim dim5(1,10,10);
  EXPECT_TRUE(dim5.is3D());
  Dim dim6(10,10,10);
  EXPECT_TRUE(dim6.is3D());
  EXPECT_FALSE(Dim::None.is3D());
}

TEST(DimTest, isEffective1D) {
  Dim dim1;
  EXPECT_TRUE(dim1.isEffective1D());
  Dim dim2(2);
  EXPECT_TRUE(dim2.isEffective1D());
  Dim dim3(1,2);
  EXPECT_TRUE(dim3.isEffective1D());
  EXPECT_TRUE(dim3.is2D());
  Dim dim4(1,1,2);
  EXPECT_TRUE(dim4.isEffective1D());
  EXPECT_TRUE(dim4.is3D());
  Dim dim5(2,1,2);
  EXPECT_FALSE(dim5.isEffective1D());
  Dim dim6(1,2,1);
  EXPECT_TRUE(dim6.isEffective1D());
  EXPECT_FALSE(Dim::None.isEffective1D());
}

TEST(DimTest, isEffective2D) {
  Dim dim1;
  EXPECT_FALSE(dim1.isEffective2D());
  Dim dim2(2);
  EXPECT_FALSE(dim2.isEffective2D());
  Dim dim3(1,2,2);
  EXPECT_TRUE(dim3.isEffective2D());
  Dim dim4(2,2,1);
  EXPECT_TRUE(dim4.isEffective2D());
  Dim dim5(2,1,2);
  EXPECT_TRUE(dim5.isEffective2D());
  Dim dim6(10,10,10);
  EXPECT_FALSE(dim6.isEffective2D());
  EXPECT_FALSE(Dim::None.isEffective2D());
}

TEST(DimTest, hasIndex) {
  Dim dim;
  EXPECT_TRUE(dim.hasIndex(Index(0,0,0)));

  dim = Dim::Cube2;
  EXPECT_TRUE(dim.hasIndex(Index(1,1,1)));
  EXPECT_FALSE(dim.hasIndex(Index(1,1,2)));

  dim = Dim::Linear256;
  EXPECT_TRUE(dim.hasIndex(Index(0)));
  EXPECT_TRUE(dim.hasIndex(Index(0,0)));
  EXPECT_TRUE(dim.hasIndex(Index(0,0,0)));
  EXPECT_TRUE(dim.hasIndex(Index(255)));
  EXPECT_TRUE(dim.hasIndex(Index(128)));
  EXPECT_TRUE(dim.hasIndex(Index(0,0,128)));
  EXPECT_FALSE(dim.hasIndex(Index(0,1,128)));
  EXPECT_FALSE(dim.hasIndex(Index(1,128)));
}

TEST(DimTest, hasLinearIndex) {
  Dim dim;
  EXPECT_TRUE(dim.hasLinearIndex(0));
  EXPECT_FALSE(dim.hasLinearIndex(1));
  EXPECT_FALSE(dim.hasLinearIndex(10));

  Dim dim2(10,10);
  EXPECT_TRUE(dim2.hasLinearIndex(99));
  EXPECT_FALSE(dim2.hasLinearIndex(100));

  Dim dim3 = Dim::Cube8;
  EXPECT_TRUE(dim3.hasLinearIndex(0));
  EXPECT_TRUE(dim3.hasLinearIndex(511));
  EXPECT_FALSE(dim3.hasLinearIndex(512));
}

TEST(DimTest, getMinIndex) {
  Dim dim;
  EXPECT_EQ(dim.getMinIndex(), Index(0));
  EXPECT_EQ(dim.getMinIndex(), Index(0,0));
  EXPECT_EQ(dim.getMinIndex(), Index(0,0));

  Dim dim2(152,12,32);
  EXPECT_EQ(dim2.getMinIndex(), Index(0));
}

TEST(DimTest, getMaxIndex) {
  Dim dim;
  EXPECT_EQ(dim.getMaxIndex(), Index(0));

  Dim dim2 = Dim::Cube8;
  EXPECT_EQ(dim2.getMaxIndex(), Index(7,7,7));
}

TEST(DimTest, getMinLinearIndex) {
  EXPECT_EQ(Dim::Linear512.getMinLinearIndex(), 0);
  EXPECT_EQ(Dim::Linear1024.getMinLinearIndex(), 0);
  EXPECT_EQ(Dim::Square512.getMinLinearIndex(), 0);
  EXPECT_EQ(Dim::Square1024.getMinLinearIndex(), 0);
  EXPECT_EQ(Dim::Cube1.getMinLinearIndex(), 0);
  EXPECT_EQ(Dim::Cube2.getMinLinearIndex(), 0);
  EXPECT_EQ(Dim::Cube4.getMinLinearIndex(), 0);
  EXPECT_EQ(Dim::Cube8.getMinLinearIndex(), 0);
}

TEST(DimTest, operatorEqual) {
  Dim dim;
  EXPECT_TRUE(dim == dim);
  EXPECT_TRUE(dim == Dim(1,1,1));
  
  Dim dimA(10,10);
  Dim dimB(10,10);
  EXPECT_TRUE(dimA == dimB);
}

TEST(DimTest, operatorNotEqual) {
  Dim dimA(10,10);
  Dim dimB(10,11);
  EXPECT_TRUE(dimA != dimB);
  EXPECT_FALSE(dimA != dimA);
}

TEST(DimTest, operatorLess) {
  Dim dimA(10,10);
  Dim dimB(10,11);
  Dim dimC(11,11);
  EXPECT_TRUE(dimA < dimB);
  EXPECT_TRUE(dimB < dimC);
  EXPECT_TRUE(dimA < dimC);
  Dim dimD(11,11);
  EXPECT_FALSE(dimC < dimD);

  Dim dimE(10,10);
  Dim dimF(100,1);
  EXPECT_FALSE(dimF < dimE);
  EXPECT_FALSE(dimF == dimE);

  Dim dimG(100);
  Dim dimH(1,100);
  EXPECT_FALSE(dimG < dimH);
  EXPECT_FALSE(dimH < dimG);
}

TEST(DimTest, operatorLessEqual) {
  EXPECT_TRUE(Dim::Cube1 <= Dim::Cube1);
  EXPECT_TRUE(Dim::Cube1 <= Dim::Cube2);
  EXPECT_FALSE(Dim::Cube4 <= Dim::Cube2);
}

TEST(DimTest, operatorGreater) {
  EXPECT_TRUE(Dim::Cube1 < Dim::Cube2);
  EXPECT_TRUE(Dim::Cube2 > Dim::Cube1);
}

TEST(DimTest, operatorGreaterEqual) {
  EXPECT_TRUE(Dim::Cube2 >= Dim::Cube2);
  EXPECT_TRUE(Dim::Cube2 >= Dim::Cube1);
}

TEST(DimTest, operatorBool) {
  Dim dim;
  EXPECT_TRUE(dim);
  EXPECT_FALSE(Dim::None);
}

TEST(DimTest, operatorIndexBrackets) {
  Dim dim;
  EXPECT_EQ(dim[0], 1);
  EXPECT_EQ(dim[1], 1);
  EXPECT_EQ(dim[2], 1);
  EXPECT_THROW(dim[15], std::out_of_range);

  EXPECT_EQ(Dim::Square1024[0], 1024);
  EXPECT_EQ(Dim::Square1024[1], 1024);
  EXPECT_EQ(Dim::Square1024[2], 1);
  EXPECT_THROW(Dim::Square1024[3], std::out_of_range);
}

TEST(DimTest, operatorIndexParen) {
  Dim dim;
  EXPECT_EQ(dim(0), 1);
  EXPECT_EQ(dim(1), 1);
  EXPECT_EQ(dim(2), 1);
  EXPECT_THROW(dim(15), std::out_of_range);

  EXPECT_EQ(Dim::Square1024(0), 1024);
  EXPECT_EQ(Dim::Square1024(1), 1024);
  EXPECT_EQ(Dim::Square1024(2), 1);
  EXPECT_THROW(Dim::Square1024(3), std::out_of_range);
}

};