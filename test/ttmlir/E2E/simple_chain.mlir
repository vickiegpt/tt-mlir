func.func @add(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = ttir.empty() : tensor<32x32xf32>
  %1 = ttir.empty() : tensor<32x32xf32>
  %2 = "ttir.add"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  %3 = "ttir.sin"(%2, %1) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %3 : tensor<32x32xf32>
}
