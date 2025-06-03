# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict
from typing import List
from ttmlir.ir import Operation
from ttmlir.dialects import func
from ttmlir.ir import Context, Module

from .enums import ExecutionType, Status


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


class OpGroup:
    def __init__(self, id):
        self.id = id
        self.ops = defaultdict(list)
        self.status = Status.PENDING

    def __setitem__(self, key: ExecutionType, value: Operation):
        self.ops[key].append(value)

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
