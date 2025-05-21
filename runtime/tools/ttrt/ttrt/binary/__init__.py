# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from ._C import (
    load_from_path,
    load_binary_from_path,
    load_binary_from_capsule,
    load_system_desc_from_path,
    verify_system_desc,
    Flatbuffer,
    GoldenTensor,
)
from . import stats

import json
import time


def as_dict(bin, logging=None):

    start_time = time.perf_counter()

    tmp = bin.as_json()

    end_time = time.perf_counter()
    elapsed = end_time - start_time
    if logging != None:
        logging.debug(f"bin.as_json() executed in {elapsed:.6f} seconds")
        logging.debug(f"type(tmp)={type(tmp)}")
        logging.debug(f"len(tmp)={len(tmp)}")

    # Flatbuffers emits 'nan' and 'inf'
    # But Python's JSON accepts only 'NaN' and 'Infinity' and nothing else
    # We include the comma to avoid replacing 'inf' in contexts like 'info'
    tmp = tmp.replace("nan,", "NaN,")
    tmp = tmp.replace("inf,", "Infinity,")

    start_time = time.perf_counter()

    result = json.loads(tmp)

    end_time = time.perf_counter()
    elapsed = end_time - start_time
    if logging != None:
        logging.debug(f"json.loads(tmp) executed in {elapsed:.6f} seconds")

    return result
