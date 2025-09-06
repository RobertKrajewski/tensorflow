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
#include <vector>

#include "tensorflow/lite/delegates/gpu/delegate_options.h"

namespace tflite {
namespace gpu {

// Local copy of the ForceFp32NodesSpec structure for testing (duplicated from delegate.cc)
struct ForceFp32NodesSpec {
  std::unordered_set<int> indices;  // Directly specified node indices
  std::vector<std::string> names;   // Operation names to be resolved later
};

// Local copy of the parsing function for testing (duplicated from delegate.cc)
ForceFp32NodesSpec ParseForceFp32NodesSpec(const char* nodes_str) {
  ForceFp32NodesSpec result;
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
          result.indices.insert(node_id);
        }
      } catch (const std::exception&) {
        // Not a number, treat as operation name
        result.names.push_back(token);
      }
    }
  }
  
  return result;
}

// Backward compatibility function - extract only indices for legacy tests
std::unordered_set<int> ParseForceFp32Nodes(const char* nodes_str) {
  return ParseForceFp32NodesSpec(nodes_str).indices;
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

// New tests for enhanced parsing with operation names
TEST(DelegatePrecisionTest, ParseForceFp32NodesSpecEmpty) {
  auto result = ParseForceFp32NodesSpec(nullptr);
  EXPECT_TRUE(result.indices.empty());
  EXPECT_TRUE(result.names.empty());
  
  result = ParseForceFp32NodesSpec("");
  EXPECT_TRUE(result.indices.empty());
  EXPECT_TRUE(result.names.empty());
  
  result = ParseForceFp32NodesSpec("   ");
  EXPECT_TRUE(result.indices.empty());
  EXPECT_TRUE(result.names.empty());
}

TEST(DelegatePrecisionTest, ParseForceFp32NodesSpecIndicesOnly) {
  auto result = ParseForceFp32NodesSpec("1,5,12");
  EXPECT_EQ(result.indices.size(), 3);
  EXPECT_TRUE(result.indices.count(1));
  EXPECT_TRUE(result.indices.count(5));
  EXPECT_TRUE(result.indices.count(12));
  EXPECT_TRUE(result.names.empty());
}

TEST(DelegatePrecisionTest, ParseForceFp32NodesSpecNamesOnly) {
  auto result = ParseForceFp32NodesSpec("conv2d_1,batch_norm_2,add_3");
  EXPECT_TRUE(result.indices.empty());
  EXPECT_EQ(result.names.size(), 3);
  EXPECT_EQ(result.names[0], "conv2d_1");
  EXPECT_EQ(result.names[1], "batch_norm_2");
  EXPECT_EQ(result.names[2], "add_3");
}

TEST(DelegatePrecisionTest, ParseForceFp32NodesSpecMixed) {
  auto result = ParseForceFp32NodesSpec("1,conv2d_1,5,batch_norm_2");
  EXPECT_EQ(result.indices.size(), 2);
  EXPECT_TRUE(result.indices.count(1));
  EXPECT_TRUE(result.indices.count(5));
  EXPECT_EQ(result.names.size(), 2);
  EXPECT_EQ(result.names[0], "conv2d_1");
  EXPECT_EQ(result.names[1], "batch_norm_2");
}

TEST(DelegatePrecisionTest, ParseForceFp32NodesSpecWithSpaces) {
  auto result = ParseForceFp32NodesSpec(" 1 , conv2d_1 , 5 , batch_norm_2 ");
  EXPECT_EQ(result.indices.size(), 2);
  EXPECT_TRUE(result.indices.count(1));
  EXPECT_TRUE(result.indices.count(5));
  EXPECT_EQ(result.names.size(), 2);
  EXPECT_EQ(result.names[0], "conv2d_1");
  EXPECT_EQ(result.names[1], "batch_norm_2");
}

TEST(DelegatePrecisionTest, ParseForceFp32NodesSpecNegativeNumbers) {
  auto result = ParseForceFp32NodesSpec("1,-5,conv2d,12");
  EXPECT_EQ(result.indices.size(), 2);  // -5 should be ignored
  EXPECT_TRUE(result.indices.count(1));
  EXPECT_TRUE(result.indices.count(12));
  EXPECT_EQ(result.names.size(), 1);
  EXPECT_EQ(result.names[0], "conv2d");
}

}  // namespace
}  // namespace gpu
}  // namespace tflite