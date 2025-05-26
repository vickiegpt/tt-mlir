// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s \
// RUN:     --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
// RUN: FileCheck --input-file=%t.mlir %s

module @maxpool2d_ceilmode attributes {} {
  func.func @test_maxpool2d_ceilmode(%arg0: tensor<1x256x56x56xbf16>) -> tensor<1x256x28x28xbf16> {
    %cst = stablehlo.constant dense<0xFF80> : tensor<bf16>
    // CHECK: "ttnn.max_pool2d"
    // CHECK-SAME: ceil_mode = true
    // CHECK-SAME: padding = array<i32: 0, 0>
    // CEHCK-SAME: -> tensor<1x1x784x256xbf16,
    %0 = "stablehlo.reduce_window"(%arg0, %cst) <{padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>}> ({
    ^bb0(%arg1: tensor<bf16>, %arg2: tensor<bf16>):
      %1 = stablehlo.maximum %arg1, %arg2 : tensor<bf16>
      stablehlo.return %1 : tensor<bf16>
    }) {ceil_mode = true, ceil_mode_padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>} : (tensor<1x256x56x56xbf16>, tensor<bf16>) -> tensor<1x256x28x28xbf16>
    return %0 : tensor<1x256x28x28xbf16>
  }
}
