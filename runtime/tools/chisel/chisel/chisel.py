# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import argparse
import pathlib
import numpy as np
import torch
import time
import os
import sys
import logging
from typing import Tuple

from ttmlir.ir import Context
from core.ops import IRModule
from core.registry import Registry
from core.golden_executor import GoldenExecutor
from core.tensors import TensorPool
from core.execution_type import ExecutionType
from utils.metrics import compute_pcc, compute_abs_err, compute_rel_err
from utils.location import parse_op_location

from ttrt.common.api import API as RtApi
from ttrt.common.util import Logger as RtLogger
from ttrt.common.util import Artifacts as RtArtifacts
from ttrt.runtime import (
    DebugHooks,
    get_op_input_refs,
    get_op_output_ref,
    get_op_debug_str,
    get_op_loc_info,
    get_tensor,
    DataType,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("chisel")

import pdb

timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
csvfile = f"pccdata_{timestamp}.csv"


class ChiselContext:
    def __init__(self, args):
        self.input_dir = args.input_dir
        self.output_dir = args.output_dir
        self.op_config = args.op_config
        self.main_fn = args.main_fn

        self.ttnn_path: pathlib.Path = (
            self.input_dir / "ttnn.mlir" if not args.ttnn_path else args.ttnn_path
        )
        self.ttir_path: pathlib.Path = (
            self.input_dir / "ttir.mlir" if not args.ttir_path else args.ttir_path
        )

        self.flatbuffer_path: pathlib.Path = (
            self.input_dir / "fb.ttnn"
            if not args.flatbuffer_path
            else args.flatbuffer_path
        )

        self.context = Context()
        self.context.load_all_available_dialects()

        logger.debug("Loading IRs...")
        self.load_irs()
        logger.debug("Setting up TTRT...")
        self.setup_ttrt()
        logger.debug("Setting up TTRT hooks...")
        self.setup_ttrt_hooks()

        if not (self.output_dir / "intermediates").exists():
            (self.output_dir / "intermediates").mkdir(parents=True, exist_ok=True)
        if not (self.output_dir / "goldens").exists():
            (self.output_dir / "goldens").mkdir(parents=True, exist_ok=True)

    def generate_inputs(self):
        pass

    def load_irs(self):
        self.ttir_module = IRModule(str(self.ttir_path.read_text()), self.context)
        self.ttnn_module = IRModule(str(self.ttnn_path.read_text()), self.context)
        self.ttir_module.set_main_op(self.main_fn)
        self.ttnn_module.set_main_op(self.main_fn)
        self.registry = Registry(self.ttir_module, self.ttnn_module)
        self.golden_tensor_pool = TensorPool()
        self.device_tensor_pool = TensorPool()
        self.planner = GoldenExecutor(self.registry, self.golden_tensor_pool)

    def setup_ttrt(self):
        args = {
            "binary": str(self.flatbuffer_path),
            "save-artifacts": True,
        }
        self.rt_logger = RtLogger()
        self.rt_artifacts = RtArtifacts(
            logger=self.rt_logger, artifacts_folder_path=str(self.output_dir)
        )
        RtApi.initialize_apis()
        self.rt_api = RtApi.Run(
            args=args, logger=self.rt_logger, artifacts=self.rt_artifacts
        )

    def compare_outputs(self, op_location: Tuple[int, int]):
        last_device_op = self.registry.get_last(op_location, ExecutionType.DEVICE)
        last_golden_op = self.registry.get_last(op_location, ExecutionType.GOLDEN)

        device_output_name = last_device_op.output.name
        golden_output_name = last_golden_op.output.name

        device_output = self.device_tensor_pool[device_output_name].data
        golden_output = self.golden_tensor_pool[golden_output_name].data

        pcc = compute_pcc(device_output, golden_output)
        abs_err = compute_abs_err(device_output, golden_output)
        rel_err = compute_rel_err(device_output, golden_output)

        print(f"PCC: {pcc}, Abs err: {abs_err}, Rel err: {rel_err}")

    def preop(self, binary, programContext, opContext):
        pass

    def postop(self, binary, programContext, opContext):
        debug_str = get_op_debug_str(opContext)
        op_location = parse_op_location(get_op_loc_info(opContext))
        # save the device output to the device tensor pool
        output_ref = get_op_output_ref(programContext, opContext)
        output_tensor = get_tensor(output_ref)

        device_op = self.registry.find_op(op_location, debug_str, ExecutionType.DEVICE)
        self.device_tensor_pool[device_op.output.name].data = output_tensor

        if self.planner.execute_golden(op_location, debug_str):
            self.compare_outputs(op_location)
        print(f"Op location: {op_location}")
        print(f"Op debug str: {debug_str}")

    def setup_ttrt_hooks(self):
        callback_env_pre = DebugHooks.get(self.preop, self.postop)

    def run(self):
        # self.run_prerequisites()
        logger.info("Running runtime...")
        result_code, results = self.rt_api()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", type=pathlib.Path, default="/localdev/ndrakulic/chisel/mnist"
    )
    parser.add_argument("--ttnn_path", type=pathlib.Path, default=None)
    parser.add_argument("--ttir_path", type=pathlib.Path, default=None)
    parser.add_argument("--flatbuffer_path", type=pathlib.Path, default=None)
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default="/localdev/ndrakulic/chisel/mnist/output",
    )
    parser.add_argument("--op_config", type=pathlib.Path)
    parser.add_argument("--main_fn", type=str, default="forward")
    args = parser.parse_args()

    chisel_context = ChiselContext(args)
    chisel_context.run()


if __name__ == "__main__":
    main()
