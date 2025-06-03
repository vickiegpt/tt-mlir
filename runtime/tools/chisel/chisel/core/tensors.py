# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from functools import cache
from typing import List, Tuple

from ttmlir.ir import Operation
from ttmlir.ir import BlockArgument, Value, Type

from .enums import ExecutionType
from ..utils.location import hash_location, parse_op_location


@dataclass
class TensorDescriptor:
    name: str
    dtype: str
    shape: List[int]
    location_hash: Tuple[int, int]
    execution_type: ExecutionType


@cache
def get_tensor_descriptor(op: Operation, execution_type: ExecutionType):
    return TensorDescriptor(
        name=op.get_name(),
        dtype=op.type.element_type.name,
        shape=op.type.shape,
        location_hash=hash_location(op.location),
        execution_type=execution_type,
    )


class TensorValue:
    def __init__(self, name: str):
        self.name = name
        self.data = None
        self.execution_data = None

    def __repr__(self) -> str:
        return f"TensorValue({self.name})"


def get_op_outputs(op: Operation, execution_type: ExecutionType):
    outputs = []
    for result in op.results:
        if hasattr(result.type, "shape") and hasattr(result.type, "element_type"):
            outputs.append(get_tensor_descriptor(result, execution_type))
    return outputs


def get_op_inputs(op: Operation, execution_type: ExecutionType):
    inputs = []
    for operand in op.operands:
        if hasattr(operand.type, "shape") and hasattr(operand.type, "element_type"):
            inputs.append(get_tensor_descriptor(operand, execution_type))
    return inputs


def get_function_inputs(op: Operation, execution_type: ExecutionType):
    inputs = []
    for arg in op.arguments:
        inputs.append(get_tensor_descriptor(arg, execution_type))
    return inputs


class TensorPool:
    def __init__(self):
        self.tensors = {}

    def __getitem__(self, key: str):
        return self.tensors[key]

    def __setitem__(self, key: str, value: TensorValue):
        self.tensors[key] = value
