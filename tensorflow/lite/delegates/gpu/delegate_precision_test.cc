/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <gtest/gtest.h>

#include <unordered_set>

#include "tensorflow/lite/delegates/gpu/delegate_options.h"

namespace tflite {
namespace gpu {
namespace {

// Expose the internal parsing function for testing
extern std::unordered_set<int> ParseForceFp32Nodes(const char* nodes_str);

TEST(DelegatePrecisionTest, ParseForceFp32NodesEmpty) {
  auto result = ParseForceFp32Nodes(nullptr);
  EXPECT_TRUE(result.empty());
  
  result = ParseForceFp32Nodes("");
  EXPECT_TRUE(result.empty());
  
  result = ParseForceFp32Nodes("   ");
  EXPECT_TRUE(result.empty());
}

TEST(DelegatePrecisionTest, ParseForceFp32NodesSingle) {
  auto result = ParseForceFp32Nodes("5");
  EXPECT_EQ(result.size(), 1);
  EXPECT_TRUE(result.count(5));
}

TEST(DelegatePrecisionTest, ParseForceFp32NodesMultiple) {
  auto result = ParseForceFp32Nodes("1,5,12");
  EXPECT_EQ(result.size(), 3);
  EXPECT_TRUE(result.count(1));
  EXPECT_TRUE(result.count(5));
  EXPECT_TRUE(result.count(12));
}

TEST(DelegatePrecisionTest, ParseForceFp32NodesWithSpaces) {
  auto result = ParseForceFp32Nodes(" 1 , 5 , 12 ");
  EXPECT_EQ(result.size(), 3);
  EXPECT_TRUE(result.count(1));
  EXPECT_TRUE(result.count(5));
  EXPECT_TRUE(result.count(12));
}

TEST(DelegatePrecisionTest, ParseForceFp32NodesInvalidEntries) {
  auto result = ParseForceFp32Nodes("1,invalid,5");
  EXPECT_EQ(result.size(), 2);
  EXPECT_TRUE(result.count(1));
  EXPECT_TRUE(result.count(5));
}

TEST(DelegatePrecisionTest, ParseForceFp32NodesNegativeNumbers) {
  auto result = ParseForceFp32Nodes("1,-5,12");
  EXPECT_EQ(result.size(), 2);  // -5 should be ignored
  EXPECT_TRUE(result.count(1));
  EXPECT_TRUE(result.count(12));
}

TEST(DelegatePrecisionTest, DefaultOptionsHaveNullForceFp32Nodes) {
  auto options = TfLiteGpuDelegateOptionsV2Default();
  EXPECT_EQ(options.force_fp32_nodes, nullptr);
}

}  // namespace
}  // namespace gpu
}  // namespace tflite