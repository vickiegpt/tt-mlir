// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/layout/from_device.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include <iostream>

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
namespace tt::runtime::ttnn::operations::layout {
void run(const ::tt::target::ttnn::FromDeviceOp *op, ProgramContext &context) {
  std::cout << "Running FromDeviceOp" << std::endl;
  ProgramTensorPool &tensorPool = context.getTensorPool();
  std::cout << "Running FromDeviceOp" << std::endl;
  const ::ttnn::Tensor &inputTensor =
      tensorPool.getTTNNTensorAndValidate(op->in());
  std::cout << "Running FromDeviceOp" << std::endl;
  DEBUG_ASSERT(!::tt::runtime::ttnn::utils::inSystemMemory(op->in()),
               "Calling ttnn::from_device on a host tensor");
  std::cout << "Running FromDeviceOp" << std::endl;
  ::ttnn::Tensor out = ::ttnn::from_device(inputTensor);
  std::cout << "Running FromDeviceOp" << std::endl;
  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
  std::cout << "Running FromDeviceOp" << std::endl;
}
} // namespace tt::runtime::ttnn::operations::layout
