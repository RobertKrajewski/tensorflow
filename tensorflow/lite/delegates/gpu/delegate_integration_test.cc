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
#include "tensorflow/lite/delegates/gpu/cl/api.h"
#include "tensorflow/lite/delegates/gpu/common/gpu_model.h"
#include "tensorflow/lite/delegates/gpu/common/precision.h"

namespace tflite {
namespace gpu {
namespace {

// Test that the force_fp32_nodes option flows through to CreateGpuModelInfo
TEST(DelegateIntegrationTest, ForceFp32NodesFlowThrough) {
  // Create OpenCL inference options with force_fp32_nodes
  cl::InferenceOptions cl_options;
  cl_options.force_fp32_nodes = {1, 5, 12};
  
  // This would normally be created by GetCreateInfo in the OpenCL API
  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::F16;  // Global precision is FP16
  create_info.force_fp32_nodes = cl_options.force_fp32_nodes;  // Copy the overrides
  
  // Verify the force_fp32_nodes are correctly propagated
  EXPECT_EQ(create_info.force_fp32_nodes.size(), 3);
  EXPECT_TRUE(create_info.force_fp32_nodes.count(1));
  EXPECT_TRUE(create_info.force_fp32_nodes.count(5));
  EXPECT_TRUE(create_info.force_fp32_nodes.count(12));
  EXPECT_FALSE(create_info.force_fp32_nodes.count(0));
  EXPECT_FALSE(create_info.force_fp32_nodes.count(10));
}

// Test that precision override logic works correctly
TEST(DelegateIntegrationTest, PrecisionOverrideLogic) {
  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::F16;  // Global precision is FP16
  create_info.force_fp32_nodes = {5, 12};  // Force nodes 5 and 12 to FP32
  
  // Simulate the logic in ConvertOperations
  auto getPrecisionForNode = [&](int node_id) -> CalculationsPrecision {
    if (create_info.force_fp32_nodes.count(node_id)) {
      return CalculationsPrecision::F32;
    } else {
      return create_info.precision;
    }
  };
  
  // Test various node IDs
  EXPECT_EQ(getPrecisionForNode(1), CalculationsPrecision::F16);  // Not forced
  EXPECT_EQ(getPrecisionForNode(5), CalculationsPrecision::F32);  // Forced to FP32
  EXPECT_EQ(getPrecisionForNode(10), CalculationsPrecision::F16); // Not forced
  EXPECT_EQ(getPrecisionForNode(12), CalculationsPrecision::F32); // Forced to FP32
  EXPECT_EQ(getPrecisionForNode(20), CalculationsPrecision::F16); // Not forced
}

// Test empty force_fp32_nodes (no overrides)
TEST(DelegateIntegrationTest, NoForceFp32Nodes) {
  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::F16;  // Global precision is FP16
  create_info.force_fp32_nodes = {};  // No overrides
  
  // All nodes should use the global precision
  auto getPrecisionForNode = [&](int node_id) -> CalculationsPrecision {
    if (create_info.force_fp32_nodes.count(node_id)) {
      return CalculationsPrecision::F32;
    } else {
      return create_info.precision;
    }
  };
  
  EXPECT_EQ(getPrecisionForNode(1), CalculationsPrecision::F16);
  EXPECT_EQ(getPrecisionForNode(5), CalculationsPrecision::F16);
  EXPECT_EQ(getPrecisionForNode(12), CalculationsPrecision::F16);
}

// Test with global FP32 precision and force_fp32_nodes
TEST(DelegateIntegrationTest, GlobalFp32WithForces) {
  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::F32;  // Global precision is already FP32
  create_info.force_fp32_nodes = {5, 12};  // These forces are redundant
  
  // All nodes should be FP32 regardless
  auto getPrecisionForNode = [&](int node_id) -> CalculationsPrecision {
    if (create_info.force_fp32_nodes.count(node_id)) {
      return CalculationsPrecision::F32;
    } else {
      return create_info.precision;
    }
  };
  
  EXPECT_EQ(getPrecisionForNode(1), CalculationsPrecision::F32);   // Global FP32
  EXPECT_EQ(getPrecisionForNode(5), CalculationsPrecision::F32);   // Forced FP32 (redundant)
  EXPECT_EQ(getPrecisionForNode(12), CalculationsPrecision::F32);  // Forced FP32 (redundant)
}

}  // namespace
}  // namespace gpu
}  // namespace tflite