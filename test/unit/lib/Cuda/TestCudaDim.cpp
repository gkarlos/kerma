#include <gtest/gtest.h>

#include <kerma/Cuda/CudaDim.h>

TEST( Constructor, Default)
{
  kerma::CudaDim dim;
  ASSERT_EQ(dim.x, 0);
  ASSERT_EQ(dim.y, 0);
  ASSERT_EQ(dim.z, 0);

}

TEST( Constructor, ThreeArg)
{
  kerma::CudaDim dim(1,2,3);
  ASSERT_EQ(dim.x, 1);
  ASSERT_EQ(dim.y, 2);
  ASSERT_EQ(dim.z, 3);
}

TEST( Constructor, Copy)
{
  kerma::CudaDim dim(1, 2, 3);
  kerma::CudaDim dim2(dim);

  ASSERT_EQ(dim.x, 1);
  ASSERT_EQ(dim.y, 2);
  ASSERT_EQ(dim.z, 3);
  ASSERT_EQ(dim.x, dim2.x);
  ASSERT_EQ(dim.y, dim2.y);
  ASSERT_EQ(dim.z, dim2.z);
  ASSERT_EQ(dim.x, 1);
  ASSERT_EQ(dim.y, 2);
  ASSERT_EQ(dim.z, 3);
}

TEST( Operator, Assign)
{
  kerma::CudaDim dim1(1,2,3);
  kerma::CudaDim dim2(4,5,6);

  ASSERT_NE(dim1.x, dim2.x);
  ASSERT_NE(dim1.y, dim2.y);
  ASSERT_NE(dim1.z, dim2.z);

  dim1 = dim2;

  ASSERT_EQ(dim1.x, dim2.x);
  ASSERT_EQ(dim1.y, dim2.y);
  ASSERT_EQ(dim1.z, dim2.z);

  ASSERT_EQ(dim1.x, 4);
  ASSERT_EQ(dim1.y, 5);
  ASSERT_EQ(dim1.z, 6);

  ASSERT_EQ(dim2.x, 4);
  ASSERT_EQ(dim2.y, 5);
  ASSERT_EQ(dim2.z, 6);
}

TEST( Operator, Compare)
{
  kerma::CudaDim dim1(1,2,3);
  kerma::CudaDim dim2(4,5,6);

  ASSERT_FALSE(dim1 == dim2);

  dim2.x = 1;
  dim2.y = 2;
  dim2.z = 3;

  ASSERT_TRUE(dim1 == dim2);

  dim2.x = 10;
  dim2.y = 11;
  dim2.z = 12;

  ASSERT_FALSE(dim1 == dim2);

  dim1 = dim2;

  ASSERT_TRUE(dim1 == dim2);
}