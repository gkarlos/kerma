#include <gtest/gtest.h>

#include "kerma/Cuda/Cuda.h"

using namespace kerma;

TEST(getCudaComputeToStr, shortDesc) {
  ASSERT_EQ( getCudaComputeToStr(CudaCompute::cc_30).compare("cc_30"), 0);
  ASSERT_EQ( getCudaComputeToStr(CudaCompute::cc_32).compare("cc_32"), 0);
  ASSERT_EQ( getCudaComputeToStr(CudaCompute::cc_35).compare("cc_35"), 0);
  ASSERT_EQ( getCudaComputeToStr(CudaCompute::Unknown).compare("Unknown"), 0);
}

TEST(getCudaComputeToStr, longDesc) {
  ASSERT_NE( getCudaComputeToStr(CudaCompute::cc_30, true).compare("cc_30"), 0);
  ASSERT_EQ( getCudaComputeToStr(CudaCompute::cc_52, true).compare("CudaCompute Capability 5.2"), 0);
  ASSERT_EQ( getCudaComputeToStr(CudaCompute::Unknown, true).compare("Unknown CudaCompute Capability"), 0);
}