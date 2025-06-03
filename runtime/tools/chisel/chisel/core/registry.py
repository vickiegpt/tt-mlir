# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict
from typing import Dict, Tuple

from ttmlir.ir import Operation
from core.tensors import TensorDescriptor, get_function_inputs, get_op_outputs
from core.execution_type import ExecutionType
from irmodule import IRModule
from utils.location import hash_location


class Registry:
    def __init__(self, ttir_module: IRModule, ttnn_module: IRModule):
        self.tensors = defaultdict(dict)
        self.tensor_to_location: Dict[ExecutionType, Dict[str, Tuple[int, int]]] = {}
        self.tensor_to_location[ExecutionType.GOLDEN] = {}
        self.tensor_to_location[ExecutionType.DEVICE] = {}
        self.op_groups = {}

        # add main fn arguments as tensors
        for arg in get_function_inputs(ttir_module.get_main_op()):
            self.add_tensor(arg, ExecutionType.GOLDEN)
        for arg in get_function_inputs(ttnn_module.get_main_op()):
            self.add_tensor(arg, ExecutionType.DEVICE)

        for op in ttir_module.dfs():
            self.add_op(op, ExecutionType.GOLDEN)
            for output in get_op_outputs(op):
                self.add_tensor(output, ExecutionType.GOLDEN)

        for op in ttnn_module.dfs():
            self.add_op(op, ExecutionType.DEVICE)
            for output in get_op_outputs(op):
                self.add_tensor(output, ExecutionType.DEVICE)

    def add_tensor(self, tensor: TensorDescriptor, kind: ExecutionType):
        self.tensor_to_location[kind][tensor.name] = tensor.location_hash
        self.tensors[tensor.location_hash][kind] = tensor

    def get_tensor(self, tensor: TensorDescriptor, kind: ExecutionType):
        return self.tensors[tensor.location_hash][kind]

    def add_op(self, op: Operation, kind: ExecutionType):
        if hash_location(op.location) not in self.op_groups:
            self.op_groups[hash_location(op.location)] = OpGroup(
                hash_location(op.location)
            )
        self.op_groups[hash_location(op.location)][kind].append(op)

    def find_op(self, location_hash: Tuple[int, int], asm: str, kind: ExecutionType):
        for op in self.op_groups[location_hash][kind]:
            if op.get_asm(enable_debug_info=True) == asm:
                return op
        return None

    def get_last(self, group_id: int, kind: ExecutionType, with_output: bool = True):
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
