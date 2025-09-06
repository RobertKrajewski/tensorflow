# Hybrid Precision GPU Delegate

This document describes the hybrid precision feature for the TensorFlow Lite GPU delegate, which allows forcing specific nodes to use FP32 precision while running others in FP16.

## Overview

The TensorFlow Lite GPU delegate typically runs in one of two modes:
- **FP32 mode**: All operations use 32-bit floating point (high precision, slower)
- **FP16 mode**: All operations use 16-bit floating point (lower precision, faster)

The hybrid precision feature allows you to selectively override specific nodes to use FP32 while the rest of the model uses FP16. This is useful when:
- You want the performance benefits of FP16 for most operations
- Certain operations are sensitive to precision loss and need FP32
- You want to fine-tune the precision/performance trade-off

## Usage

### Basic Setup

```cpp
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/delegates/gpu/delegate_options.h"

// Create delegate options
TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();

// Enable precision loss globally (allows FP16)
options.is_precision_loss_allowed = 1;

// Force specific nodes to use FP32 (comma-separated node indices or operation names)
options.force_fp32_nodes = "5,12,23";  // By node index
// OR
options.force_fp32_nodes = "CONV_2D,ADD";  // By operation name  
// OR  
options.force_fp32_nodes = "5,CONV_2D,12,ADD";  // Mixed format

// Create and use the delegate
auto* delegate = TfLiteGpuDelegateV2Create(&options);
// ... use with TensorFlow Lite interpreter ...
TfLiteGpuDelegateV2Delete(delegate);
```

### Node Specification

You can specify nodes to force to FP32 in three ways:

#### 1. By Node Index
Node indices correspond to the execution order in your TensorFlow Lite model:
```cpp
options.force_fp32_nodes = "5,12,23";  // Force nodes at indices 5, 12, and 23
```

#### 2. By Operation Name  
Specify operations by their TensorFlow Lite operation name:
```cpp
options.force_fp32_nodes = "CONV_2D,ADD";  // Force all CONV_2D and ADD operations
```

#### 3. Mixed Format
Combine both approaches:
```cpp
options.force_fp32_nodes = "5,CONV_2D,12,ADD";  // Mixed indices and operation names
```

### Identifying Nodes

**By Index**: You can identify node indices by:
1. **Model visualization**: Use tools like Netron to visualize your model  
2. **TensorFlow Lite analysis**: Use debugging tools to trace execution
3. **Empirical testing**: Run with different precision settings and measure accuracy

**By Operation Name**: Common TensorFlow Lite operation names include:
- `CONV_2D` - 2D convolution operations
- `DEPTHWISE_CONV_2D` - Depthwise convolution operations  
- `FULLY_CONNECTED` - Dense/linear layers
- `ADD`, `MUL`, `SUB` - Element-wise arithmetic operations
- `AVERAGE_POOL_2D`, `MAX_POOL_2D` - Pooling operations
- `BATCH_NORMALIZATION` - Batch normalization layers
- `RELU`, `RELU6`, `LOGISTIC` - Activation functions

**Advantages of Each Approach**:
- **Index-based**: Precise control over specific operation instances
- **Name-based**: Easy to target all operations of a certain type
- **Mixed**: Best of both worlds for complex targeting strategies

### Common Use Cases

**Classification models**: Force the final layers to FP32
```cpp
// By index (precise control)
options.force_fp32_nodes = "45,46,47";  // Final dense + softmax layers

// By operation type (applies to all instances)
options.force_fp32_nodes = "FULLY_CONNECTED,SOFTMAX";

// Mixed approach 
options.force_fp32_nodes = "45,SOFTMAX";  // Specific dense layer + all softmax
```

**Object detection**: Force critical detection layers to FP32
```cpp
// By index for specific detection heads
options.force_fp32_nodes = "12,25,38";  

// By operation type for all convolutions in detection heads
options.force_fp32_nodes = "CONV_2D";

// Target specific arithmetic operations that affect bounding boxes
options.force_fp32_nodes = "ADD,MUL";  // Box coordinate calculations
```

**Batch normalization**: Force precision-sensitive normalization
```cpp
// By index for specific batch norm layers
options.force_fp32_nodes = "8,15,22,29";  

// By operation type for all batch normalization layers
options.force_fp32_nodes = "BATCH_NORMALIZATION";

// Target batch norm in specific parts of the network
options.force_fp32_nodes = "45,BATCH_NORMALIZATION,FULLY_CONNECTED";
```

## Technical Details

### Implementation

The feature works by:
1. Parsing the `force_fp32_nodes` string into node indices and operation names
2. During delegate preparation, resolving operation names to specific node indices using the TensorFlow Lite context
3. Propagating this information through the OpenCL compilation pipeline  
4. Overriding the `OperationDef.precision` field for specified nodes during model building
5. Ensuring proper cache invalidation for serialized models

### Performance Considerations

- **Memory**: FP32 operations use 2x the memory of FP16 operations
- **Bandwidth**: FP32 tensors require 2x the memory bandwidth
- **Compute**: FP32 operations may be slower on some GPU architectures
- **Cache**: Different precision configurations generate different compiled kernels

### Limitations

- Only works with OpenCL backend (not OpenGL)
- Node indices are model-specific and may change if the model is modified
- Mixing precisions may introduce conversion overhead between operations
- Not all GPU architectures have the same FP16/FP32 performance characteristics

## Testing and Validation

### Accuracy Testing

Compare your hybrid precision model against:
1. **Full FP32**: Should have similar accuracy to your forced FP32 nodes
2. **Full FP16**: Should have better accuracy than pure FP16
3. **Original model**: Should maintain acceptable accuracy while improving performance

### Performance Testing

Measure:
- **Inference latency**: End-to-end inference time
- **Memory usage**: Peak GPU memory consumption  
- **Throughput**: Inferences per second for batch processing

### Example Test

```cpp
// Test different configurations
auto testConfig = [](const char* force_nodes) {
    TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
    options.is_precision_loss_allowed = 1;
    options.force_fp32_nodes = force_nodes;
    
    auto* delegate = TfLiteGpuDelegateV2Create(&options);
    // ... run inference and measure accuracy/performance ...
    TfLiteGpuDelegateV2Delete(delegate);
};

testConfig(nullptr);           // Pure FP16
testConfig("5,12,23");        // Hybrid precision
testConfig("");               // Pure FP16 (empty string)
```

## Migration Guide

### From Existing Code

If you're currently using:
```cpp
options.is_precision_loss_allowed = 0;  // Pure FP32
```

You can migrate to:
```cpp
options.is_precision_loss_allowed = 1;   // Enable FP16
options.force_fp32_nodes = "critical_node_ids";  // Keep critical nodes in FP32
```

### Identifying Critical Nodes

1. Start with pure FP16: `options.is_precision_loss_allowed = 1;`
2. Measure accuracy degradation compared to FP32
3. Systematically add nodes back to FP32 using `force_fp32_nodes`
4. Find the minimal set that maintains acceptable accuracy

## Examples

See the following files for complete examples:
- `delegate_usage_example.cc`: Basic usage patterns
- `delegate_precision_test.cc`: Unit tests for parsing functionality
- `delegate_integration_test.cc`: Integration tests for the complete feature

## Troubleshooting

### Serialization Issues
If you encounter cache invalidation problems, ensure your `force_fp32_nodes` string is consistent between runs.

### Performance Regression
If hybrid precision is slower than pure FP16:
- Reduce the number of forced FP32 nodes
- Profile to identify bottlenecks
- Consider whether the accuracy gain justifies the performance cost

### Compilation Errors
Make sure you have:
- OpenCL backend enabled
- Valid node indices (within the model's node count)
- Proper delegate lifecycle management