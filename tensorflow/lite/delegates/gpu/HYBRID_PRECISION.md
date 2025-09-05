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

// Force specific nodes to use FP32 (comma-separated node indices)
options.force_fp32_nodes = "5,12,23";

// Create and use the delegate
auto* delegate = TfLiteGpuDelegateV2Create(&options);
// ... use with TensorFlow Lite interpreter ...
TfLiteGpuDelegateV2Delete(delegate);
```

### Node Index Identification

Node indices correspond to the execution order in your TensorFlow Lite model. You can identify them by:

1. **Model visualization**: Use tools like Netron to visualize your model
2. **TensorFlow Lite analysis**: Use debugging tools to trace execution
3. **Empirical testing**: Run with different precision settings and measure accuracy

### Common Use Cases

**Classification models**: Force the final layers to FP32
```cpp
options.force_fp32_nodes = "45,46,47";  // Final dense + softmax layers
```

**Object detection**: Force critical detection layers to FP32
```cpp
options.force_fp32_nodes = "12,25,38";  // Key detection heads
```

**Batch normalization**: Force precision-sensitive normalization
```cpp
options.force_fp32_nodes = "8,15,22,29";  // Batch norm layers
```

## Technical Details

### Implementation

The feature works by:
1. Parsing the `force_fp32_nodes` string into a set of node IDs
2. Propagating this information through the OpenCL compilation pipeline
3. Overriding the `OperationDef.precision` field for specified nodes during model building
4. Ensuring proper cache invalidation for serialized models

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