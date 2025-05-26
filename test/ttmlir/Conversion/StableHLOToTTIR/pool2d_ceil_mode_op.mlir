// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<1x256x56x56xbf16>) -> tensor<1x256x28x28xbf16> {
    %cst = stablehlo.constant dense<0xFF80> : tensor<bf16>
    // CHECK: %[[POOLING1:[0-9]+]] = "ttir.pooling"(%arg0, %{{[0-9]+}})
    // CHECK-SAME: ceil_mode = true
    // CHECK-SAME: ceil_mode_padding = array<i64: 0, 0, 0, 0, 1, 1, 1, 1>
    %0 = "stablehlo.reduce_window"(%arg0, %cst) <{padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>}> ({
    ^bb0(%arg1: tensor<bf16>, %arg2: tensor<bf16>):
      %1 = stablehlo.maximum %arg1, %arg2 : tensor<bf16>
      stablehlo.return %1 : tensor<bf16>
    }) {ceil_mode = true, ceil_mode_padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>} : (tensor<1x256x56x56xbf16>, tensor<bf16>) -> tensor<1x256x28x28xbf16>
    // CHECK: return %[[POOLING1]] :
    // CHECK-SAME: tensor<1x256x28x28xbf16>
    return %0 : tensor<1x256x28x28xbf16>
  }
}
