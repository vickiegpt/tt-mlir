# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from typing import List, Enum

from ttmlir.ir import (
    Context,
    Module,
    Operation,
    WalkOrder,
    WalkResult,
    BlockArgument,
    Location,
    Value,
)
from ttmlir.dialects import func

import pdb


@dataclass
class TensorDescriptor:
    name: str
    dtype: str
    shape: list[int]
    location: Location = None
    arg: BlockArgument | Value = None


def hash_tensor(location: Location):
    assert location is not None
    return (location.line, location.column)


def get_op_outputs(op: Operation):
    outputs = []
    for result in op.results:
        outputs.append(
            TensorDescriptor(
                name=result.get_name(),
                dtype=result.type.element_type,
                shape=result.type.shape,
                location=result.location,
                arg=result,
            )
        )
    return outputs


def get_op_inputs(op: Operation):
    inputs = []
    for operand in op.operands:
        inputs.append(
            TensorDescriptor(
                name=operand.get_name(),
                dtype=operand.type.element_type,
                shape=operand.type.shape,
                location=operand.location,
                arg=operand,
            )
        )
    return inputs


def get_function_inputs(op: Operation):
    inputs = []
    for arg in op.arguments:
        inputs.append(
            TensorDescriptor(
                arg.get_name(), arg.type.element_type, arg.type.shape, arg.location, arg
            )
        )
    return inputs


def get_function_outputs(op: Operation):
    NotImplementedError()


class IRModule:
    def __init__(self, mlir_text: str, context: Context):
        self.mlir_module = Module.parse(mlir_text, context)
        self.context = context
        self._main_op: Operation | None = None

    def set_main_op(self, name: str):
        for op in self.dfs(self.mlir_module.operation, WalkOrder.POST_ORDER):
            if isinstance(op, func.FuncOp) and op.name.value == name:
                self._main_op = op
                return

    def get_main_op(self):
        return self._main_op

    def dfs(
        self, op: Operation | None = None, walk_order: WalkOrder = WalkOrder.POST_ORDER
    ):
        if op is None:
            op = self._main_op
            assert op is not None
        ops = []

        def _walk_ops(op):
            nonlocal ops
            ops.append(op.opview)
            return WalkResult.ADVANCE

        op.operation.walk(_walk_ops, walk_order=walk_order)
        return ops


class Type(Enum):
    DEVICE = "device"
    GOLDEN = "golden"


class Registry:
    def __init__(self):
        self.tensors = defaultdict(defaultdict(list))
        self.op_groups = defaultdict(defaultdict(list))

    def add_tensor(self, tensor: TensorDescriptor, kind: Type):
        self.tensors[hash_tensor(tensor.location)][kind].append(tensor)

    def get_tensor(self, tensor: TensorDescriptor, kind: Type):
        return self.tensors[hash_tensor(tensor.location)][kind]

    def add_op(self, op: Operation, kind: Type):
        self.op_groups[hash_tensor(op.location)][kind].append(op)

    def get_op(self, op: Operation, kind: Type):
        return self.op_groups[hash_tensor(op.location)][kind]

    def get_last(self, group_id: int, kind: Type, with_output: bool = True):
        if with_output:
            for op in self.op_groups[group_id][kind][::-1]:
                if len(op.results) > 0:
                    return op.results[0]
        else:
            return self.op_groups[group_id][kind][-1]


if __name__ == "__main__":
    ttir_path = Path("/localdev/ndrakulic/chisel/mnist/ttir.mlir")
    ttnn_path = Path("/localdev/ndrakulic/chisel/mnist/ttnn.mlir")

    test_path = ttnn_path
    # registry = Registry()
    # module = IRModule(test_path.read_text(), Context())
    # module.set_main_op("forward")

    # # Going through all the ops in the module
    # for op in module.dfs():
    #     print(op.name)

    # # Get inputs to the main function
    # main_op = module.get_main_op()
    # function_inputs = get_function_inputs(main_op)
    # print(function_inputs)

    # print("--------------------------------")
    # print("Inputs and Outputs")
    # print("--------------------------------")

    # for op in module.dfs():
    #     print(op.name)
    #     print("--------------------------------")
    #     print("Inputs:")
    #     print(*get_op_inputs(op), sep="\n")
    #     print("Outputs:")
    #     print(*get_op_outputs(op), sep="\n")
    #     print("--------------------------------")
