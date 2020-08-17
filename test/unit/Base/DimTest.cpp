#include "kerma/Base/Dim.h"

#include "gtest/gtest.h"

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

};