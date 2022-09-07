#pragma once

#include <gtest/gtest.h>

#include <tdnat/ops.h>

class LLVMEnvironment : public testing::Environment {
public:
  ~LLVMEnvironment() override {}
  void SetUp() override { tdnat::initialize_llvm(); }
};

class RegistryEnvironment : public testing::Environment {
public:
  ~RegistryEnvironment() override {}
  void SetUp() override { tdnat::initialize(); }
};

static const auto *env =
#ifdef TDNAT_LOAD_REGISTRY
    testing::AddGlobalTestEnvironment(new RegistryEnvironment());
#else
    testing::AddGlobalTestEnvironment(new LLVMEnvironment());
#endif
