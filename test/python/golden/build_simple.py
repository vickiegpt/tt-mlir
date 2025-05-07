# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from ttmlir.test_utils import compile_to_flatbuffer
from ttmlir.ttir_builder import Operand, TTIRBuilder, Shape

import os


def big_golden(shape: Shape):
    def simple_test(in0: Operand, builder: TTIRBuilder):
        return builder.cos(in0)

    compile_to_flatbuffer(
        simple_test,
        [shape],
        test_base="big_golden",
        system_desc_path=os.environ["SYSTEM_DESC_PATH"],
    )


if __name__ == "__main__":
    input_shape = (1, 64, 1024, 1024)
    big_golden(input_shape)
