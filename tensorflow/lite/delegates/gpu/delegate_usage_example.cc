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

void ExampleUsageWithOperationNames() {
  // Create delegate options
  TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
  
  // Allow precision loss globally (enables FP16)
  options.is_precision_loss_allowed = 1;
  
  // Force all convolution and batch normalization operations to FP32
  // This is useful when you want to target operation types rather than specific indices
  options.force_fp32_nodes = "CONV_2D,BATCH_NORMALIZATION";
  
  // Create the delegate
  auto* delegate = TfLiteGpuDelegateV2Create(&options);
  
  // ... rest of TensorFlow Lite inference setup ...
  
  // Clean up
  TfLiteGpuDelegateV2Delete(delegate);
}

void ExampleUsageWithMixedFormat() {
  // Create delegate options
  TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
  
  // Allow precision loss globally (enables FP16)
  options.is_precision_loss_allowed = 1;
  
  // Mixed format: force specific node indices and all operations of certain types
  // This forces node 5, all CONV_2D operations, node 12, and all ADD operations to FP32
  options.force_fp32_nodes = "5,CONV_2D,12,ADD";
  
  // Create the delegate
  auto* delegate = TfLiteGpuDelegateV2Create(&options);
  
  // ... rest of TensorFlow Lite inference setup ...
  
  // Clean up
  TfLiteGpuDelegateV2Delete(delegate);
}

/*
Usage Notes:
============

1. Node Specification Methods:
   a) Node indices: "5,12,23" - Comma-separated node indices  
   b) Operation names: "CONV_2D,ADD" - Comma-separated operation names
   c) Mixed format: "5,CONV_2D,12,ADD" - Combination of indices and names

2. Node indices correspond to the order of operations in the TensorFlow Lite model.

3. Common TensorFlow Lite operation names include:
   - CONV_2D, DEPTHWISE_CONV_2D - Convolution operations
   - FULLY_CONNECTED - Dense/linear layers  
   - ADD, MUL, SUB - Element-wise arithmetic
   - BATCH_NORMALIZATION - Batch normalization
   - RELU, RELU6, LOGISTIC - Activation functions
   - AVERAGE_POOL_2D, MAX_POOL_2D - Pooling operations

4. You can identify critical nodes by:
   - Running the model with FP16 and comparing accuracy
   - Using TensorFlow Lite's model analyzer tools
   - Examining the model architecture for precision-sensitive operations

5. Common operations that might benefit from FP32:
   - Final classification layers (FULLY_CONNECTED, SOFTMAX)
   - Batch normalization operations (BATCH_NORMALIZATION)
   - Operations with small gradients or large dynamic ranges

6. The feature works with serialization:
   - The cache key includes force_fp32_nodes information
   - Models are re-compiled when the force_fp32_nodes setting changes
   - This ensures correctness across different precision configurations

Example node identification by index:
====================================
If you have a model with this structure:
0: INPUT
1: CONV2D 
2: RELU
3: CONV2D
4: RELU  
5: BATCH_NORMALIZATION  <- Force to FP32
6: CONV2D
7: RELU
8: FULLY_CONNECTED     <- Force to FP32
9: SOFTMAX             <- Force to FP32
10: OUTPUT

You could use any of these approaches:
- By index: options.force_fp32_nodes = "5,8,9";
- By operation: options.force_fp32_nodes = "BATCH_NORMALIZATION,FULLY_CONNECTED,SOFTMAX";
- Mixed: options.force_fp32_nodes = "5,FULLY_CONNECTED,SOFTMAX";
*/