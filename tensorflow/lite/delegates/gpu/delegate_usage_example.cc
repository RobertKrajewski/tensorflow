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

// Example usage of the hybrid precision GPU delegate feature
//
// This example demonstrates how to force specific nodes to use FP32 precision
// while allowing other nodes to use FP16 for better performance.

#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/delegates/gpu/delegate_options.h"

void ExampleUsage() {
  // Create delegate options
  TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
  
  // Allow precision loss globally (enables FP16)
  options.is_precision_loss_allowed = 1;
  
  // Force specific nodes (e.g., nodes 5, 12, and 23) to use FP32 precision
  // This is useful for operations that are sensitive to precision loss
  options.force_fp32_nodes = "5,12,23";
  
  // Create the delegate
  auto* delegate = TfLiteGpuDelegateV2Create(&options);
  
  // ... rest of TensorFlow Lite inference setup ...
  // interpreter->ModifyGraphWithDelegate(delegate);
  
  // Clean up
  TfLiteGpuDelegateV2Delete(delegate);
}

/*
Usage Notes:
============

1. The force_fp32_nodes option takes a comma-separated string of node indices.
2. Node indices correspond to the order of operations in the TensorFlow Lite model.
3. You can identify critical nodes by:
   - Running the model with FP16 and comparing accuracy
   - Using TensorFlow Lite's model analyzer tools
   - Examining the model architecture for precision-sensitive operations

4. Common operations that might benefit from FP32:
   - Final classification layers
   - Batch normalization operations
   - Operations with small gradients or large dynamic ranges

5. The feature works with serialization:
   - The cache key includes force_fp32_nodes information
   - Models are re-compiled when the force_fp32_nodes setting changes
   - This ensures correctness across different precision configurations

Example node identification:
===========================
If you have a model with this structure:
0: INPUT
1: CONV2D 
2: RELU
3: CONV2D
4: RELU  
5: BATCH_NORM  <- Force to FP32
6: CONV2D
7: RELU
8: FULLY_CONNECTED  <- Force to FP32
9: SOFTMAX  <- Force to FP32
10: OUTPUT

You would use: options.force_fp32_nodes = "5,8,9";
*/