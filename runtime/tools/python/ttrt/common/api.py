# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import json
import importlib.machinery
import sys
import signal
import os
import io
import subprocess
import time
import socket
from pkg_resources import get_distribution
import sys
import shutil
import atexit

from ttrt.common.util import *


def randn(shape, dtype):
    import torch

    return torch.randn(shape, dtype=dtype)


def arange(shape, dtype):
    import torch

    def volume(shape):
        v = 1
        for i in shape:
            v *= i
        return v

    return torch.arange(volume(shape), dtype=dtype).reshape(shape)


init_fns_map = {
    "randn": randn,
    "arange": arange,
}
init_fns = sorted(list(init_fns_map.keys()))


def check_identity(args, binary_name, torch_inputs, torch_outputs):
    import torch

    for i, o in zip(torch_inputs, torch_outputs):
        if not torch.allclose(i, o, rtol=args.rtol, atol=args.atol):
            print(
                "Failed: inputs and outputs do not match in binary",
                binary_name,
            )
            print(torch.abs(i - o))
        else:
            print("Passed:", binary_name)


#######################################################################################
########################################**API**########################################
#######################################################################################

"""
API: version
  - get version of ttrt
"""


def version(args):
    package_name = "ttrt"
    try:
        package_version = get_distribution(package_name).version
    except Exception as e:
        print(f"Error retrieving version: {e} for {package_name}")
    print(package_version)


"""
API: read
  - read contents of flatbuffer file
"""


def read(args):
    # initialization
    binaries = []
    fbb_list = []

    # acquire parameters
    arg_binary = args.binary
    arg_clean_artifacts = args.clean_artifacts
    arg_save_artifacts = args.save_artifacts
    arg_section = args.section

    # preprocessing
    if os.path.isdir(arg_binary):
        print("provided directory of flatbuffers")
        binaries = find_ttnn_files(arg_binary)
    else:
        binaries.append(arg_binary)

    if arg_clean_artifacts:
        print("cleaning artifacts")
        clean_artifacts()

    if arg_save_artifacts:
        print("setting up artifact directories")
        setup_artifacts(binaries)

    # constraint checking
    print("executing constraint for all provided flatbuffers")
    for binary in binaries:
        check_file_exists(binary)
        fbb = ttrt.binary.load_from_path(binary)
        check_version(fbb.version)
        fbb_list.append(fbb)

    # execution
    print("executing action for all provided flatbuffers")
    for fbb in fbb_list:
        read_actions[arg_section](fbb)

    # save artifacts
    if arg_save_artifacts:
        print("saving artifacts")
        for binary in binaries:
            copy_ttnn_binary_into_artifact(binary)


"""
API: run
  - run flatbuffer on device
"""


def run(args):
    import ttrt.runtime
    import torch

    # initialization
    binaries = []
    fbb_list = []
    torch_inputs = {}
    torch_outputs = {}
    system_desc = None

    # acquire parameters
    arg_binary = args.binary
    arg_program_index = args.program_index
    arg_clean_artifacts = args.clean_artifacts
    arg_loops = int(args.loops)
    arg_save_artifacts = args.save_artifacts

    # preprocessing
    if os.path.isdir(arg_binary):
        print("provided directory of flatbuffers")
        binaries = find_ttnn_files(arg_binary)
    else:
        binaries.append(arg_binary)

    if arg_clean_artifacts:
        print("cleaning artifacts")
        clean_artifacts()

    if arg_save_artifacts:
        print("setting up artifact directories")
        setup_artifacts(binaries)

    # constraint checking
    print("executing constraint for all provided flatbuffers")
    system_desc, device_ids = ttrt.runtime.get_current_system_desc()
    program_indices = []
    cleaned_binaries = []
    for binary in binaries:
        check_file_exists(binary)
        fbb = ttrt.binary.load_binary_from_path(binary)
        check_version(fbb.version)
        fbb_dict = ttrt.binary.as_dict(fbb)

        if fbb_dict["system_desc"] != system_desc_as_dict(system_desc)["system_desc"]:
            print(
                f"system descriptor for binary and system mismatch, ignoring test={binary}"
            )
            continue

        if arg_program_index != "all":
            program_index = int(arg_program_index)
            assert program_index < len(
                fbb_dict["programs"]
            ), "args.program_index out of range"
            program_indices.append(program_index)
        else:
            program_indices = [i for i in range(len(fbb_dict["programs"]))]

        fbb_list.append(
            (
                os.path.splitext(os.path.basename(binary))[0],
                fbb,
                fbb_dict,
                program_indices,
            )
        )
        cleaned_binaries.append(binary)
    binaries = cleaned_binaries

    # execution
    print("executing action for all provided flatbuffers")
    # TODO: sort the flatbuffers by runtime type, for now just take the first one
    ttrt.runtime.set_compatible_runtime(fbb_list[0][1])
    device = ttrt.runtime.open_device([device_ids[0]])
    atexit.register(lambda: ttrt.runtime.close_device(device))

    for bin in binaries:
        self.logging.info(f"evaluating binary={bin.file_path}")

        program_indices = []
        if self["program_index"] == "all":
            program_indices.extend(range(bin.get_num_programs()))
        else:
            program_indices.append(int(self["program_index"]))

        for program_index in program_indices:
            self.logging.debug(
                f"evaluating program={program_index} for binary={bin.file_path}"
            )

            program = bin.get_program(program_index)
            program.populate_inputs(
                API.Run.TorchInitilizer.get_initilizer(self["init"])
            )
            program.populate_outputs(
                API.Run.TorchInitilizer.get_initilizer("zeros")
            )

            total_inputs = []
            total_outputs = []
            for loop in range(self["loops"]):
                self.logging.debug(
                    f"generating inputs/outputs for loop={loop+1}/{self['loops']} for binary={bin.file_path}"
                )

                inputs = []
                outputs = []
                for i in program.input_tensors:
                    inputs.append(
                        ttrt.runtime.create_tensor(
                            i.data_ptr(),
                            list(i.shape),
                            list(i.stride()),
                            i.element_size(),
                            Binary.Program.to_data_type(i.dtype),
                        )
                    )

                for i in program.output_tensors:
                    outputs.append(
                        ttrt.runtime.create_tensor(
                            i.data_ptr(),
                            list(i.shape),
                            list(i.stride()),
                            i.element_size(),
                            Binary.Program.to_data_type(i.dtype),
                        )
                    )

                total_inputs.append(inputs)
                total_outputs.append(outputs)

        event = None
        for loop in range(arg_loops):
            event = ttrt.runtime.submit(
                device, fbb, program_index, total_inputs[loop], total_outputs[loop]
            )
            print(f"finished loop={loop}")
        ttrt.runtime.wait(event)
        print("outputs:\n", torch_outputs)
        if args.identity:
            check_identity(
                args,
                binary_name,
                torch_inputs[binary_name][program_index],
                torch_outputs[binary_name][program_index],
            )

# save artifacts
if arg_save_artifacts:
    print("saving artifacts")
    for binary in binaries:
        fbb_dict = ttrt.binary.as_dict(ttrt.binary.load_binary_from_path(binary))
        curr_program_indices = []

        if arg_program_index != "all":
            curr_program_indices.append(int(arg_program_index))
        else:
            curr_program_indices = [i for i in range(len(fbb_dict["programs"]))]

        for program_index in curr_program_indices:
            copy_ttnn_binary_into_artifact(binary)
            binary_name = os.path.splitext(os.path.basename(binary))[0]
            torch_input_tensors = torch_inputs[binary_name][program_index]
            torch_output_tensors = torch_outputs[binary_name][program_index]

            for i, input in enumerate(torch_input_tensors):
                save_torch_tensor_into_ttrt_artifacts(
                    input, f"{binary_name}/program_{program_index}_input_{i}.pt"
                )

            for i, output in enumerate(torch_output_tensors):
                save_torch_tensor_into_ttrt_artifacts(
                    output, f"{binary_name}/program_{program_index}_output_{i}.pt"
                )

            save_system_desc_into_ttrt_artifacts(
                system_desc, f"{binary_name}/system_desc.ttsys"
            )


"""
API: query
  - query device for system descriptor in the form of a flatbuffer
"""


def query(args):
    import ttrt.runtime

    # initialization
    system_desc = None

    # acquire parameters
    arg_system_desc = args.system_desc
    arg_system_desc_as_json = args.system_desc_as_json
    arg_system_desc_as_dict = args.system_desc_as_dict
    arg_clean_artifacts = args.clean_artifacts
    arg_save_artifacts = args.save_artifacts

    # preprocessing
    if arg_clean_artifacts:
        print("cleaning artifacts")
        clean_artifacts()

    if arg_save_artifacts:
        print("setting up artifact directories")
        setup_artifacts()

    # execution
    print("executing action to get system desc")
    system_desc = ttrt.runtime.get_current_system_desc()[0]

    if arg_system_desc or arg_system_desc_as_json:
        print(system_desc.as_json())
    if arg_system_desc_as_dict:
        print(system_desc_as_dict(system_desc))

    # save artifacts
    if arg_save_artifacts:
        print("saving artifacts")
        save_system_desc_into_ttrt_artifacts(system_desc, "system_desc.ttsys")


"""
API: perf
  - run flatbuffer on device in performance mode
"""


def perf(args):
    import ttrt.common.perf_trace as perf_trace

    # initialization
    binaries = []

    # acquire parameters
    arg_binary = args.binary
    arg_program_index = args.program_index
    arg_clean_artifacts = args.clean_artifacts
    arg_perf_csv = args.perf_csv
    arg_loops = int(args.loops)
    arg_save_artifacts = args.save_artifacts
    arg_device = args.device
    arg_generate_params = args.generate_params

    # preprocessing
    if os.path.isdir(arg_binary):
        print("provided directory of flatbuffers")
        binaries = find_ttnn_files(arg_binary)
    else:
        binaries.append(arg_binary)

    if arg_clean_artifacts:
        print("cleaning artifacts")
        clean_artifacts()

    if arg_save_artifacts:
        print("setting up artifact directories")
        setup_artifacts(binaries)

    # constraint checking
    if arg_generate_params:
        check_file_exists(arg_perf_csv)

    for binary in binaries:
        check_file_exists(binary)

    # execution
    if arg_generate_params:
        perf_trace.generate_params_dict(arg_perf_csv)
    else:
        for binary in binaries:
            # get available port for tracy client and server to communicate on
            port = perf_trace.get_available_port()

            if not port:
                raise Exception("No available port found")

            env_vars = dict(os.environ)
            self.globals.add_global_env("TRACY_PORT", port)
            self.globals.add_global_env(
                "TT_METAL_DEVICE_PROFILER_DISPATCH", "0"
            )

            if self["device"]:
                self.globals.add_global_env("TT_METAL_DEVICE_PROFILER", "1")

            tracy_capture_tool_command = f"{self.tracy_capture_tool_path} -o {PROFILER_LOGS_DIR / TRACY_FILE_NAME} -f -p {port}"
            self.tracy_capture_tool_process = subprocess.Popen(
                tracy_capture_tool_command, shell=True
            )

            upstream_apis = API.Run.get_upstream_apis()
            command_options = ""

            for api in upstream_apis:
                name = (
                    api["name"]
                    if not api["name"].startswith("-")
                    else api["name"].lstrip("-")
                )
                name = name.replace("-", "_")

                if api["type"] == bool:
                    if self[name]:
                        command_options += f" {api['name']} "
                else:
                    command_options += f" {api['name']} {self[name]} "

            test_command = f"python -m tracy -p {self.globals.get_ttmlir_venv_path()}/bin/ttrt run {bin.file_path} --save-artifacts {command_options}"
            print(f"test command for binary={bin.file_path} is: {test_command}")
            testProcess = subprocess.Popen(
                [test_command], shell=True, env=env_vars, preexec_fn=os.setsid
            )

            def signal_handler(sig, frame):
                os.killpg(os.getpgid(testProcess.pid), signal.SIGTERM)
                self.tracy_capture_tool_process.terminate()
                self.tracy_capture_tool_process.communicate()
                sys.exit(3)

            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            testProcess.communicate()

            try:
                captureProcess.communicate(timeout=15)
                perf_trace.generate_report(binary_perf_folder)
            except subprocess.TimeoutExpired as e:
                captureProcess.terminate()
                captureProcess.communicate()
                print(
                    f"No profiling data could be captured. Please make sure you are on the correct build. Use scripts/build_scripts/build_with_profiler_opt.sh to build if you are not sure."
                )
                sys.exit(1)

            # save artifacts
            perf_trace.save_perf_artifacts(binary_perf_folder)
