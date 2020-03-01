#include "gtest/gtest.h"

#include "kerma/Cuda/NVVM.h"

#include <iostream>

using namespace kerma;

TEST(AddressSpace, Defaults)
{
  ASSERT_EQ( AddressSpace::CODE.getName().compare("code"), 0);
  ASSERT_EQ( AddressSpace::GENERIC.getName().compare("generic"), 0);
  ASSERT_EQ( AddressSpace::GLOBAL.getName().compare("global"), 0);
  ASSERT_EQ( AddressSpace::SHARED.getName().compare("shared"), 0);
  ASSERT_EQ( AddressSpace::CONSTANT.getName().compare("constant"), 0);
  ASSERT_EQ( AddressSpace::LOCAL.getName().compare("local"), 0);
  ASSERT_EQ( AddressSpace::UNKNOWN.getName().compare("unknown"), 0);

  ASSERT_EQ( AddressSpace::CODE.getCode(), 0);
  ASSERT_EQ( AddressSpace::GENERIC.getCode(), 0);
  ASSERT_EQ( AddressSpace::GLOBAL.getCode(), 1);
  ASSERT_EQ( AddressSpace::SHARED.getCode(), 3);
  ASSERT_EQ( AddressSpace::CONSTANT.getCode(), 4);
  ASSERT_EQ( AddressSpace::LOCAL.getCode(), 5);
  ASSERT_EQ( AddressSpace::UNKNOWN.getCode(), -1);
}

TEST(AddressSpace, Creation)
{
  AddressSpace a("test", 15);
  ASSERT_EQ( a.getName().compare("test"), 0);
  ASSERT_EQ( a.getCode(), 15);
}

TEST(AddressSpace, OverloadedOperators) 
{
  AddressSpace a("code", 0);
  AddressSpace b("global", 1);

  ASSERT_TRUE(a == AddressSpace::CODE);
  ASSERT_TRUE(b == AddressSpace::GLOBAL);
  ASSERT_TRUE( a != b);
  ASSERT_TRUE(AddressSpace::GLOBAL == AddressSpace::GLOBAL);
  ASSERT_TRUE( AddressSpace::GLOBAL != AddressSpace::LOCAL);
}