# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import Callable, List

from ttir_builder import Operand, TTIRBuilder, Shape, TypeInfo
from ttir_builder.utils import compile_to_flatbuffer, Marks, shape_str
from ttmlir.dialects import stablehlo
from ttmlir.ir import (
    DenseI64ArrayAttr,
    DenseI32ArrayAttr,
)


@pytest.mark.parametrize(
    "shapes,batch_dims_lhs,contract_dims_lhs,batch_dims_rhs,contract_dims_rhs",
    [
        (
            # [(4, 10, 3, 5, 7), (4, 10, 5, 7, 3), (4, 10, 3, 7, 10, 7, 3)],
            # [(1, 2, 3, 4), (1, 2, 4, 3), (1, 2, 3, 2, 3)],
            # [(8,), (8,), (1,)],
            [(2, 2, 2), (2, 2, 2), (2, 2, 2, 2)],
            [],
            [1],
            [],
            [2],
            # [],[0],[],[0],
        )
    ],
)
def test_dot_general(
    shapes: List[Shape],
    batch_dims_lhs: List[int],
    contract_dims_lhs: List[int],
    batch_dims_rhs: List[int],
    contract_dims_rhs: List[int],
    request,
):
    def dot_general(
        in0: Operand,
        in1: Operand,
        out0: Operand,
        builder: TTIRBuilder,
        unit_attrs: List[str] = None,
    ):
        return builder.dot_general(
            in0,
            in1,
            out0,
            batch_dims_lhs,
            contract_dims_lhs,
            batch_dims_rhs,
            contract_dims_rhs,
            unit_attrs=unit_attrs,
        )

    compile_to_flatbuffer(
        dot_general,
        shapes,
        # [torch.int8, torch.int8, torch.int8],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )
    assert False, "test"
