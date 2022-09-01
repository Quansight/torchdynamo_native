#pragma once

#include <gtest/gtest.h>

#include <tdnat/ops.h>

class Environment : public testing::Environment {
public:
  ~Environment() override {}
  void SetUp() override { tdnat::initialize_llvm(); }
};

// Initializing LLVM once for each test file.
static const auto *env = testing::AddGlobalTestEnvironment(new Environment());
