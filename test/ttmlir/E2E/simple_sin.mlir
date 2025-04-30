func.func @add(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = ttir.empty() : tensor<32x32xf32>
  %1 = "ttir.sin"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %1 : tensor<32x32xf32>
}
