// RUN: ttmlir-opt --tt-register-device="system-desc-path=%system_desc_path%" --ttnn-decompose-layouts %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %basename_t.ttnn
// RUN: ttmlir-opt --ttnn-modify-signatures-for-dylib --convert-ttnn-to-emitc %t.mlir > %t2.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir > %basename_t.cpp
//
// This test tests to_memory_config op. It runs a single "ttnn.to_layout" that transfers a tensor from dram to l1, hence it gets decomposed into ttnn.to_memory_config.
//
// Line below checks that to_memory_config op appears after TTIR to TTNN conversion.
// RUN: FileCheck %s --input-file %t.mlir
#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>
#ttnn_layout_device_tile_dram = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!tt.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout_device_tile_l1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!tt.tile<32x32, f32>, #l1>, <interleaved>>

func.func @to_memory_config_from_dram_to_l1(%arg0: tensor<64x128xf32, #ttnn_layout_device_tile_dram>) -> tensor<64x128xf32, #ttnn_layout_device_tile_l1> {
  %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.to_layout"(%arg0, %0) <{dtype = #tt.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#l1, <<2x4>>, <interleaved>>}> : (tensor<64x128xf32, #ttnn_layout_device_tile_dram>, !ttnn.device) -> tensor<64x128xf32, #ttnn_layout_device_tile_l1>
  // CHECK: "ttnn.to_memory_config"
  // CHECK-SAME: memory_config = #ttnn.memory_config<#l1_, <<2x4>>, <interleaved>>
  return %1 : tensor<64x128xf32, #ttnn_layout_device_tile_l1>
}