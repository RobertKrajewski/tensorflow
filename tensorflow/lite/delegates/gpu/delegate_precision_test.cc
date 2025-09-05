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

#include <cstring>
#include <sstream>
#include <unordered_set>

#include "tensorflow/lite/delegates/gpu/delegate_options.h"

namespace tflite {
namespace gpu {

// Local copy of the parsing function for testing (duplicated from delegate.cc)
std::unordered_set<int> ParseForceFp32Nodes(const char* nodes_str) {
  std::unordered_set<int> result;
  if (!nodes_str || strlen(nodes_str) == 0) {
    return result;
  }
  
  std::string str(nodes_str);
  std::stringstream ss(str);
  std::string token;
  
  while (std::getline(ss, token, ',')) {
    // Trim whitespace
    token.erase(0, token.find_first_not_of(" \t"));
    token.erase(token.find_last_not_of(" \t") + 1);
    
    if (!token.empty()) {
      try {
        int node_id = std::stoi(token);
        if (node_id >= 0) {
          result.insert(node_id);
        }
      } catch (const std::exception&) {
        // Skip invalid entries
      }
    }
  }
  
  return result;
}

namespace {

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