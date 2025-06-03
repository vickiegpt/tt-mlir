# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from functools import cache
from collections import defaultdict
from typing import List

from ttmlir.ir import Operation, WalkOrder, WalkResult, Context, Module
from ttmlir.dialects import func

from .enums import ExecutionType, Status
from .tensors import TensorDescriptor, get_op_inputs, get_op_outputs


class Op:
    def __init__(self, op: Operation, execution_type: ExecutionType):
        self.inputs: List[TensorDescriptor] = get_op_inputs(op, execution_type)
        self.outputs: List[TensorDescriptor] = get_op_outputs(op, execution_type)
        self.name = op.name
        self.location = op.location
        self.asm = op.get_asm()
        self.execution_type = execution_type
        self.ir_op = op
        self.args = None


@cache
def get_op(op: Operation, execution_type: ExecutionType):
    return Op(op, execution_type)


class OpGroup:
    def __init__(self, id):
        self.id = id
        self.ops = defaultdict(list)
        self.status = Status.PENDING

    def __getitem__(self, key: ExecutionType):
        return self.ops[key]

    def __len__(self):
        return len(self.ops)

    def items(self):
        return self.ops.items()

    def get_last(self, kind: ExecutionType, with_output: bool = True):
        if with_output:
            for op in self.ops[kind][::-1]:
                if len(op.results) > 0:
                    return op
        else:
            return self.ops[kind][-1]


class IRModule:
    def __init__(self, mlir_text: str, context: Context, execution_type: ExecutionType):
        self.mlir_module = Module.parse(mlir_text, context)
        self.context = context
        self._main_op: Operation | None = None
        self.op_groups = {}
        self.execution_type = execution_type

    def set_main_op(self, name: str):
        for op in self._dfs(self.mlir_module.operation):
            if isinstance(op, func.FuncOp) and op.name.value == name:
                self._main_op = op
                return

    def get_main_op(self):
        return self._main_op

    def main_body_ops(self, op: Operation | None = None) -> List[Operation]:
        if op is None:
            op = self._main_op
            assert op is not None
        ops = []
        for region in op.regions:
            for block in region.blocks:
                for op in block.operations:
                    if op.name == "ttir.empty":
                        continue
                    ops.append(get_op(op, self.execution_type))
        return ops

    def create_op_groups(self):
        for op in self.main_body_ops():
            if op.location_hash not in self.op_groups:
                self.op_groups[op.location_hash] = OpGroup(op.location_hash)
            self.op_groups[op.location_hash].append(op)

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
