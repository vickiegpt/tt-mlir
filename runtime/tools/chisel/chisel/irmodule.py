# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from typing import List, Tuple
from enum import Enum
import torch

from ttmlir.ir import (
    Context,
    Module,
    Operation,
    WalkOrder,
    WalkResult,
    BlockArgument,
    Location,
    Value,
    Type,
)
from ttmlir.dialects import func

from mapping import ttir_to_torch_mapping, ttir_dtype_maps

import pdb


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


def hash_location(location: Location):
    assert location is not None
    return (location.start_line, location.start_col)


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


class IRModule:
    def __init__(self, mlir_text: str, context: Context):
        self.mlir_module = Module.parse(mlir_text, context)
        self.context = context
        self._main_op: Operation | None = None

    def set_main_op(self, name: str):
        for op in self._dfs(self.mlir_module.operation):
            if isinstance(op, func.FuncOp) and op.name.value == name:
                self._main_op = op
                return

    def get_main_op(self):
        return self._main_op

    def dfs(self, op: Operation | None = None) -> List[Operation]:
        if op is None:
            op = self._main_op
            assert op is not None
        ops = []
        for region in op.regions:
            for block in region.blocks:
                for op in block.operations:
                    if op.name == "ttir.empty":
                        continue
                    ops.append(op.opview)
        return ops

    def _dfs(
        self, op: Operation | None = None, walk_order: WalkOrder = WalkOrder.POST_ORDER
    ):
        if op is None:
            op = self._main_op
            assert op is not None
        ops = []

        def _walk_ops(op):
            nonlocal ops
            if not op.name == "ttir.empty":
                ops.append(op.opview)
            return WalkResult.ADVANCE

        op.operation.walk(_walk_ops, walk_order=walk_order)
        return ops


class Type(Enum):
    DEVICE = "device"
    GOLDEN = "golden"


class Status(Enum):
    PENDING = "pending"
    EXECUTED = "executed"


class OpGroup:
    def __init__(self, id):
        self.id = id
        self.ops = defaultdict(list)
        self.status = Status.PENDING

    def __setitem__(self, key: Type, value: Operation):
        self.ops[key].append(value)

    def __getitem__(self, key: Type):
        return self.ops[key]

    def __len__(self):
        return len(self.ops)

    def items(self):
        return self.ops.items()

    def get_last(self, kind: Type, with_output: bool = True):
        if with_output:
            for op in self.ops[kind][::-1]:
                if len(op.results) > 0:
                    return op
        else:
            return self.ops[kind][-1]


class Registry:
    def __init__(self, ttir_module: IRModule, ttnn_module: IRModule):
        self.tensors = defaultdict(dict)
        self.tensor_to_location: Dict[Type, Dict[str, Tuple[int, int]]] = {}
        self.tensor_to_location[Type.GOLDEN] = {}
        self.tensor_to_location[Type.DEVICE] = {}
        self.op_groups = {}

        # add main fn arguments as tensors
        for arg in get_function_inputs(ttir_module.get_main_op()):
            self.add_tensor(arg, Type.GOLDEN)
        for arg in get_function_inputs(ttnn_module.get_main_op()):
            self.add_tensor(arg, Type.DEVICE)

        for op in ttir_module.dfs():
            self.add_op(op, Type.GOLDEN)
            for output in get_op_outputs(op):
                self.add_tensor(output, Type.GOLDEN)

        for op in ttnn_module.dfs():
            self.add_op(op, Type.DEVICE)
            for output in get_op_outputs(op):
                self.add_tensor(output, Type.DEVICE)

    def add_tensor(self, tensor: TensorDescriptor, kind: Type):
        self.tensor_to_location[kind][tensor.name] = tensor.location_hash
        self.tensors[tensor.location_hash][kind] = tensor

    def get_tensor(self, tensor: TensorDescriptor, kind: Type):
        return self.tensors[tensor.location_hash][kind]

    def add_op(self, op: Operation, kind: Type):
        if hash_location(op.location) not in self.op_groups:
            self.op_groups[hash_location(op.location)] = OpGroup(
                hash_location(op.location)
            )
        self.op_groups[hash_location(op.location)][kind].append(op)

    def find_op(self, location_hash: Tuple[int, int], asm: str, kind: Type):
        for op in self.op_groups[location_hash][kind]:
            if op.get_asm(enable_debug_info=True) == asm:
                return op
        return None

    def get_last(self, group_id: int, kind: Type, with_output: bool = True):
        return self.op_groups[group_id].get_last(kind, with_output)

    def print(self):
        print("\n" * 2)
        print("--------------------------------")
        print("Op Groups")
        print("--------------------------------")
        for group_id in sorted(self.op_groups):
            op_groups = self.op_groups[group_id]
            print(f"Group {group_id}:")
            for kind, ops in op_groups.items():
                print(f"\t{kind}:")
                for op in ops:
                    print(f"\t\t{op.name} {hash_location(op.location)}")

        print("\n" * 2)
        print("--------------------------------")
        print("Tensors")
        print("--------------------------------")
        for group_id, tensors in self.tensors.items():
            print(f"Group {group_id}:")
            for kind, tensor in tensors.items():
                print(f"\t{kind}: {tensor.name} {tensor.type}")


class TensorPool:
    def __init__(self):
        self.tensors = {}

    def __getitem__(self, key: str):
        return self.tensors[key]

    def __setitem__(self, key: str, value: TensorValue):
        self.tensors[key] = value


class GoldenExecutor:
    def __init__(self, registry: Registry, golden_tensor_pool: TensorPool):
        self.registry = registry
        self.last_golden_executed = None
        # sorted location list
        self.op_locations = sorted(self.registry.op_groups.keys())
        self.loc_iter = iter(self.op_locations)
        self.golden_tensor_pool = golden_tensor_pool

    def execute(self, op: Operation):
        print(f"Executing operation: {op.name}")
        print(f"Operation ASM: {op.get_asm()}")
        print(f"Operation location: {op.location}")
        print(f"Executing op: {op.name}")

        if op.name not in ttir_to_torch_mapping:
            # TODO: enable to enter debug mode so you can add on the fly mapping if missing
            raise ValueError(f"Unknown op: {op.name}")

        mapping = ttir_to_torch_mapping[op.name]

        outputs = get_op_outputs(op)
        inputs = [
            self.golden_tensor_pool[input.name]
            for input in get_op_inputs(op)
            if input.name not in outputs
            and hasattr(input.arg.owner, "name")
            and input.arg.owner.name != "ttir.empty"
        ]
        print(
            f"Input shapes: {[(inp.name, x.golden_data.shape if x is not None else None) for inp, x in zip(get_op_inputs(op), inputs)]}"
        )
        op_result = mapping(op, inputs)
        if op.name == "func.return":
            return op_result

        for output in outputs:
            tensor_name = output.name
            if op_result is not None:
                print(f"Output shape: {tensor_name} = {op_result.shape}")
            self.golden_tensor_pool[tensor_name] = op_result

        return op_result

    def execute_golden(self, device_op_location: Tuple[int, int], op_asm: str) -> bool:
        last_device_op = self.registry.get_last(device_op_location, Type.DEVICE)
        if last_device_op is None:
            print(f"No last device op found for location {device_op_location}")
            return False
        if last_device_op.get_asm(enable_debug_info=True) != op_asm:
            print(f"ASM mismatch at {device_op_location}")
            print(f"Expected: {op_asm}")
            print(f"Got: {last_device_op.get_asm(enable_debug_info=True)}")
            return False

        to_execute = []
        for loc in self.loc_iter:
            if loc <= device_op_location:
                to_execute.append(loc)
            if loc >= device_op_location:
                break

        print(f"Executing golden ops from groups: {to_execute}")
        for loc in to_execute:
            for op in self.registry.op_groups[loc][Type.GOLDEN]:
                self.execute(op)
        return True


if __name__ == "__main__":
    model = "llama_debug"
    model = "mnist"
    fn_name = "forward"
    ttir_path = Path(f"/localdev/ndrakulic/chisel/{model}/ttir.mlir")
    ttnn_path = Path(f"/localdev/ndrakulic/chisel/{model}/ttnn.mlir")

    context = Context()

    ttnn_module = IRModule(str(ttnn_path.read_text()), context)
    ttnn_module.set_main_op(fn_name)
    ttir_module = IRModule(str(ttir_path.read_text()), context)
    ttir_module.set_main_op(fn_name)

    registry = Registry(ttir_module, ttnn_module)
    golden_tensor_pool = TensorPool()
    planner = GoldenExecutor(registry, golden_tensor_pool)

    for arg in get_function_inputs(ttir_module.get_main_op()):
        name = arg.name
        shape = arg.type.shape
        device_dtype = arg.type.element_type
        golden_dtype = ttir_dtype_maps[str(arg.type.element_type)]
        print(
            f"Adding tensor {name} with shape {shape} and dtype {device_dtype} to golden tensor pool"
        )
        tensor_value = TensorValue(name)
        tensor_value.golden_data = torch.ones(shape, dtype=golden_dtype)
        golden_tensor_pool[name] = tensor_value

    registry.print()

    for op in ttnn_module.dfs():
        op_location = hash_location(op.location)
        op_asm = op.get_asm(enable_debug_info=True)
        print(f"Executing golden ops up to location {op_location}, OP_NAME: {op.name}")
        planner.execute_golden(op_location, op_asm)
        print(
            f"Finished executing golden ops up to location {op_location}, OP_NAME: {op.name}"
        )
        print("-" * 40)
    registry.print()
