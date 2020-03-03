#include <gtest/gtest.h>

#include "kerma/Cuda/Cuda.h"

using namespace kerma;

TEST( isSupportedCudaCompute, __ ) {
  ASSERT_TRUE(isSupportedCudaCompute(CudaCompute::cc_30));
  ASSERT_TRUE(isSupportedCudaCompute(CudaCompute::cc_52));
  ASSERT_TRUE(isSupportedCudaCompute(CudaCompute::cc_72));
  ASSERT_TRUE(isSupportedCudaCompute(CudaCompute::Unknown));
  ASSERT_FALSE(isSupportedCudaCompute(CudaCompute::cc_10));
  ASSERT_FALSE(isSupportedCudaCompute(CudaCompute::cc_20));
  ASSERT_FALSE(isSupportedCudaCompute(CudaCompute::cc_21));
  ASSERT_FALSE(isSupportedCudaCompute(CudaCompute::cc_80));
}

TEST( isSupportedCudaArch, __ ) {
  ASSERT_TRUE(isSupportedCudaArch(CudaArch::sm_30));
  ASSERT_TRUE(isSupportedCudaArch(CudaArch::sm_52));
  ASSERT_TRUE(isSupportedCudaArch(CudaArch::sm_72));
  ASSERT_TRUE(isSupportedCudaArch(CudaArch::Unknown));
  ASSERT_FALSE(isSupportedCudaArch(CudaArch::sm_10));
  ASSERT_FALSE(isSupportedCudaArch(CudaArch::sm_20));
  ASSERT_FALSE(isSupportedCudaArch(CudaArch::sm_21));
  ASSERT_FALSE(isSupportedCudaArch(CudaArch::sm_80));
}

TEST(getCudaComputeStr, shortDesc) {
  ASSERT_EQ( getCudaComputeStr(CudaCompute::cc_30).compare("cc_30"), 0);
  ASSERT_EQ( getCudaComputeStr(CudaCompute::cc_32).compare("cc_32"), 0);
  ASSERT_EQ( getCudaComputeStr(CudaCompute::cc_35).compare("cc_35"), 0);
  ASSERT_EQ( getCudaComputeStr(CudaCompute::Unknown).compare("Unknown"), 0);
}

TEST(getCudaComputeStr, longDesc) {
  ASSERT_NE( getCudaComputeStr(CudaCompute::cc_30, true).compare("cc_30"), 0);
  ASSERT_EQ( getCudaComputeStr(CudaCompute::cc_52, true).compare("CudaCompute Capability 5.2"), 0);
  ASSERT_EQ( getCudaComputeStr(CudaCompute::Unknown, true).compare("Unknown CudaCompute Capability"), 0);
}

TEST(getCudaArchStr, __ ) {
  ASSERT_EQ( getCudaArchStr(CudaArch::sm_50).compare("sm_50"), 0);
  ASSERT_EQ( getCudaArchStr(CudaArch::sm_52).compare("sm_52"), 0);
  ASSERT_NE( getCudaArchStr(CudaArch::sm_32).compare("sm_30"), 0);
  ASSERT_NE( getCudaArchStr(CudaArch::sm_75).compare("Unknown Architecture"), 0);
  ASSERT_EQ( getCudaArchStr(CudaArch::Unknown).compare("Unknown Architecture"), 0);
}

TEST(getCudaArchName, CudaArch ) {
  ASSERT_EQ( getCudaArchName(CudaArch::sm_30).compare( getCudaArchName(CudaArch::sm_32)), 0);
  ASSERT_EQ( getCudaArchName(CudaArch::sm_52).compare( getCudaArchName(CudaArch::sm_53)), 0);
  ASSERT_EQ( getCudaArchName(CudaArch::Unknown).compare("Unknown"), 0);
  ASSERT_NE( getCudaArchName(CudaArch::sm_72).compare( "Unknown"), 0);
  ASSERT_EQ( getCudaArchName(CudaArch::sm_50).compare("Maxwell"), 0);
  ASSERT_EQ( getCudaArchName(CudaArch::sm_70).compare("Volta"), 0);
  ASSERT_EQ( getCudaArchName(CudaArch::sm_75).compare("Turing"), 0);
}

TEST(getCudaArchName, CudaCompute) {
  ASSERT_EQ( getCudaArchName(CudaCompute::cc_30).compare( getCudaArchName(CudaCompute::cc_32)), 0);
  ASSERT_EQ( getCudaArchName(CudaCompute::cc_52).compare( getCudaArchName(CudaCompute::cc_53)), 0);
  ASSERT_EQ( getCudaArchName(CudaCompute::Unknown).compare("Unknown"), 0);
  ASSERT_NE( getCudaArchName(CudaCompute::cc_72).compare( "Unknown"), 0);
  ASSERT_EQ( getCudaArchName(CudaCompute::cc_50).compare("Maxwell"), 0);
  ASSERT_EQ( getCudaArchName(CudaCompute::cc_70).compare("Volta"), 0);
  ASSERT_EQ( getCudaArchName(CudaCompute::cc_75).compare("Turing"), 0);
}

TEST(getCudaArch, string)
{
  ASSERT_EQ( getCudaArch("sm_30"), CudaArch::sm_30);
  ASSERT_EQ( getCudaArch("sm_52"), CudaArch::sm_52);
  ASSERT_NE( getCudaArch("sm_80"), CudaArch::sm_80);
  ASSERT_EQ( getCudaArch("Unknown"), CudaArch::Unknown);
  ASSERT_EQ( getCudaArch("test"), CudaArch::Unknown);
}

TEST(getCudaArch, numeric)
{
  ASSERT_EQ( getCudaArch(3.0), CudaArch::sm_30);
  ASSERT_EQ( getCudaArch(30), CudaArch::sm_30);
  ASSERT_EQ( getCudaArch(5.2), CudaArch::sm_52);
  ASSERT_NE( getCudaArch(8.0), CudaArch::sm_80);
  ASSERT_EQ( getCudaArch(200.2), CudaArch::Unknown);
}

TEST(getCudaSideStr, __ )
{
  ASSERT_EQ( getCudaSideStr(CudaSide::HOST).compare("host"), 0);
  ASSERT_EQ( getCudaSideStr(CudaSide::DEVICE).compare("device"), 0);
  ASSERT_EQ( getCudaSideStr(CudaSide::Unknown).compare("unknown"), 0);
}

TEST(getCudaSide, __ )
{
  ASSERT_EQ( getCudaSide("host"), CudaSide::HOST);
  ASSERT_EQ( getCudaSide("Host"), CudaSide::HOST);
  ASSERT_EQ( getCudaSide("HOST"), CudaSide::HOST);
  ASSERT_EQ( getCudaSide("Device"), CudaSide::DEVICE);
  ASSERT_EQ( getCudaSide("device"), CudaSide::DEVICE);
  ASSERT_EQ( getCudaSide("DEVICE"), CudaSide::DEVICE);
  ASSERT_EQ( getCudaSide(""), CudaSide::Unknown);
  ASSERT_EQ( getCudaSide("cudaside"), CudaSide::Unknown);
}