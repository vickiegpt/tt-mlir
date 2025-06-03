# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from functools import cache
from typing import Tuple

from ttmlir.ir import Operation
from ttmlir.ir import BlockArgument, Value, Type
from core.execution_type import ExecutionType
from utils.location import hash_location, parse_op_location


@dataclass
class TensorDescriptor:
    name: str
    type: Type
    location_hash: Tuple[int, int] = None
    arg: BlockArgument | Value = None


class TensorValue:
    def __init__(self, name: str):
        self.name = name
        self.data = None
        self.execution_data = None

    def __repr__(self) -> str:
        return f"TensorValue({self.name})"


def get_op_outputs(op: Operation):
    outputs = []
    for result in op.results:
        if hasattr(result.type, "shape") and hasattr(result.type, "element_type"):
            outputs.append(
                TensorDescriptor(
                    name=result.get_name(),
                    type=result.type,
                    location_hash=hash_location(result.location),
                    arg=result,
                )
            )
    return outputs


def get_op_inputs(op: Operation):
    inputs = []
    for operand in op.operands:
        if hasattr(operand.type, "shape") and hasattr(operand.type, "element_type"):
            inputs.append(
                TensorDescriptor(
                    name=operand.get_name(),
                    type=operand.type,
                    location_hash=hash_location(operand.location),
                    arg=operand,
                )
            )
    return inputs


def get_function_inputs(op: Operation):
    inputs = []
    for arg in op.arguments:
        inputs.append(
            TensorDescriptor(
                name=arg.get_name(),
                type=arg.type,
                location_hash=hash_location(arg.location),
                arg=arg,
            )
        )
    return inputs


class TensorPool:
    def __init__(self):
        self.tensors = {}

    def __getitem__(self, key: str):
        return self.tensors[key]

    def __setitem__(self, key: str, value: TensorValue):
        self.tensors[key] = value
