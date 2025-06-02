# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
```
class RawIRModule {
  +mlir_module : mlir.ir.Module      «exact PyMLIR object»
  +type        : {'golden'|'device'}
  +context
}
```

| Signature                           | Purpose / Notes                                                                      |
| ----------------------------------- | ------------------------------------------------------------------------------------ |
| `walk_ops(*, nested: bool = False)` | Generator that yields `mlir.ir.Operation`s; delegates to `.body.walk()` if `nested`. |
| `_dfs`                              | go through all modules                                                               |
| set_main_function                   | set the `_main_opertaion` based on name                                              |
| get_main_function                   | return `_main_operation`                                                             |

```
class GroupRegistry {
  +groups : dict[(int,int), Group]
  +get_group(k) -> Group
}
```

| Signature                                | Purpose                                                                             |
| ---------------------------------------- | ----------------------------------------------------------------------------------- |
| `add_op(op: Op, kind)`                   | Looks at `op.loc` → key; adds to (or creates) the `Group` also to the correct kind. |
| `get_group(key: tuple[int,int]) -> Group | None`                                                                               |
| `iter_groups()`                          | Yields `Group`s sorted by line                                                      |
| `__contains__(key)`                      | `'if (line,col) in registry'`.                                                      |

```
class AnnotatedModule {
  +raw     : RawIRModule
  +reg     : GroupRegistry
  +ops     : list[Op]
  +tensors : dict[str, TensorHandle]
  +register()
}
```

|Scope|Signature|Purpose|
|---|---|---|
|core|`register()`|Already planned—walk ops, create `Op`, create/lookup `TensorHandle`, call `GroupRegistry.add_op`.|
|core|`_get_or_create_tensor(ir_val, side)`|Internal helper used by `register()`.|
|convenience|`get_tensor_by_id(tid) -> TensorHandle|None`|
|convenience|`get_group_for_tensor(tid)`||
|convenience|`iter_groups()`|Proxy to registry but limited to groups that contain this module’s ops.|
|dunder|`__len__` → number of ops.||


```
class Op {
  +mlir_op : mlir.ir.Operation
  +loc     : (line,col)
  +inputs  : list[TensorHandle]
  +outputs : list[TensorHandle]
}
```

| Signature                                                    | Purpose                                          |
| ------------------------------------------------------------ | ------------------------------------------------ |
| `add_input(handle: TensorHandle)` / `add_output(handle)`     | Makes relationship explicit; updates both lists. |
| `is_last_in_group(group: Group, with_output)`                | Used by hooks.                                   |
| `signature()`                                                | String `"add(%0,%1) -> %2"`.                     |
| `__hash__` (keyed by `mlir_op`) → lets you use `Op` in sets. | get_asm                                          |

```
class TensorHandle {
  +id              : str         «stable across both sides»
  +golden_val      : mlir.ir.Value|None  «producer value in TTIR»
  +device_val      : mlir.ir.Value|None  «producer value in TTNN»
  +golden_data     : torch.Tensor|None
  +device_data     : RuntimeTensor|None
  +execution_data  : property
}
```

| Signature                                                         | Notes                                                                |
| ----------------------------------------------------------------- | -------------------------------------------------------------------- |
| `set_golden_val(val: ir.Value)` / `set_device_val(val: ir.Value)` | Attach MLIR producers after parsing.                                 |
| `set_golden_data(t: torch.Tensor)` / `set_device_data(rt_tensor)` | Already half-implemented in the big code dump → keep same semantics. |

```
class CPUEngine {
  +run(ops, feeds) -> dict[tensor_id, torch.Tensor]
}
```

| Scope       | Signature                                                                             | Purpose                                           |
| ----------- | ------------------------------------------------------------------------------------- | ------------------------------------------------- |
| core        | `run_ops(op_seq: list[Op], feeds: dict[str, torch.Tensor]) -> dict[id, torch.Tensor]` | Thin wrapper; uses `ttir_executor` mapping table. |
| convenience | `warmup()`                                                                            | Optional JIT/perf warm-up.                        |

```
class RunSession {
  +registry        : GroupRegistry
  +golden_mod      : AnnotatedModule
  +device_mod      : AnnotatedModule
  +cpu_engine      : CPUEngine
  +run()
  +_pre_hook(ctx)
  +_post_hook(ctx, outs)
}
```

|Scope|Signature|Purpose|
|---|---|---|
|core|`run()`|Compile (if needed) → register hooks → execute device main().|
|core|`_install_hooks(rt)`|Single place to bind `self._pre_hook/_post_hook`.|
|core|`_is_group_boundary(op_ctx) -> bool`|Called in post-hook to know when to launch CPU side.|
|core|`_pre_hook(binary, pc, oc)` / `_post_hook(...)`|Already sketched.|
|convenience|`flush()`|Placeholder for future ArtifactLogger flush.|
|convenience|`reset()`|Re-run on a fresh registry.|
|dunder|`__enter__/__exit__`|Allow `with RunSession(cfg) as sess:` pattern.|

```
class Group {
  +id          : (line,col)
  +golden_ops  : list[Op]
  +device_ops  : list[Op]
  +inputs      : list[TensorHandle]
  +outputs     : list[TensorHandle]
  +status      : {'pending','done','skipped'}
}
```

| Signature                                                  | Purpose                                                  |
| ---------------------------------------------------------- | -------------------------------------------------------- |
| `finalize_io()`                                            | Compute `.inputs` / `.outputs` after all ops registered. |
| `mark_done()` / `mark_skipped()`                           | Flip `.status`; maybe timestamp.                         |
| `get_last_golden_op(with_output=False)` (already there)    |                                                          |
| `iter_ops(side: Literal["golden","device","both"]="both")` |                                                          |
| `summary()`                                                | Returns small dict for CSV/log.                          |


## 9 · Utility helpers (module-level, not class methods)

|Helper|Why|
|---|---|
|`location_to_key(loc) -> (int,int)`|Centralise “line/col extraction” so everyone shares behaviour.|
|`tensor_id_from_value(val) -> str`|Canonicalises SSA names (`"%123"` → `"123"` or leave as is).|
|`is_mlir_op_has_result(op) -> bool`|Saves repeated `hasattr(op,"results")` checks.|

"""
from enum import Enum
from typing import Set
from pathlib import Path
import pdb

from ttmlir.ir import Module, Context, Operation, Value, Location, WalkResult, WalkOrder

from ttrt.runtime import DebugHooks


def location2key(loc: Location | str):
    if isinstance(loc, str):
        raise NotImplementedError()
    if isinstance(loc, Location):
        return (loc.start_line, loc.start_col)

    raise NotImplementedError(f"Unknown location type: {type(loc)}")


def get_op_output(op: Operation):
    if hasattr(op, "results"):
        if len(op.results) == 0:
            return None
        if len(op.results) > 1:
            raise NotImplementedError(f"Op has more than one result: {op}")
        return op.results[0]
    if hasattr(op, "output"):
        return op.output
    if hasattr(op, "outputs"):
        return op.outputs[0]
    return None


class Type(Enum):
    DEVICE = "device"
    GOLDEN = "golden"


class RawIRModule:
    def __init__(self, mlir_text: str, ctx: Context):
        self.mlir_module = Module.parse(mlir_text, ctx)
        self.ctx = ctx
        self._main_op: Operation | None = None

    def dfs(
        self, op: Operation | None = None, walk_order: WalkOrder = WalkOrder.POST_ORDER
    ):
        if op is None:
            op = self.mlir_module.operation
        ops = []

        def _walk_ops(op):
            nonlocal ops
            ops.append(op)
            return WalkResult.ADVANCE

        op.operation.walk(_walk_ops, walk_order=walk_order)
        print("DFS:")
        print(*ops, sep="\n")
        return ops

    def main_body(self, op: Operation | None = None):
        if op is None:
            op = self._main_op
            assert op is not None
        for region in op.regions:
            for block in region.blocks:
                for inner in block.operations:
                    yield inner

    def set_main_op(self, name: str):
        for op in self.dfs(self.mlir_module.operation, WalkOrder.POST_ORDER):
            print(op.name)
            if op.name == "func.func" and str(op.attributes["sym_name"]) == f'"{name}"':
                self._main_op = op
                return
            if hasattr(op.name, "value") and op.name.value == name:
                self._main_op = op
                return
        print(f"Could not find main op: {name}")

    def get_inputs(self):
        assert self._main_op is not None

        if hasattr(self._main_op, "arguments"):
            return self._main_op.arguments
        inputs = self._main_op.attributes["function_type"].value.inputs
        return {f"%arg{i}": val for i, val in enumerate(inputs)}


class AnnotatedModule:
    def __init__(self, ir_module: RawIRModule, registry: "Registry", kind: Type):
        self.ir_module = ir_module
        self.registry = registry
        self.kind = kind

    def register(self):
        for op in self.ir_module.dfs():
            self.registry.add_op(op, self.kind)


class TensorHandle:
    def __init__(self, golden_val: Value | None, device_val: Value | None):
        self.golden_val = golden_val
        self.device_val = device_val
        self.golden_data = None
        self.device_data = None
        self.execution_data = None

    def check_shape(self):
        NotImplementedError()

    def __repr__(self):
        golden_str = (
            "None"
            if self.golden_val is None
            else f"{self.golden_val.get_name()}: {self.golden_val.type}"
        )


class Op:
    def __init__(self, op: Operation, location: Location):
        self.op = op
        self.loc = location
        self.inputs = []
        self.outputs = []


class OpGroup:
    def __init__(self, id, registry: "Registry"):
        self.id = id
        self.ops: dict[Type, list[Op]] = {
            Type.GOLDEN: [],
            Type.DEVICE: [],
        }
        self.inputs: Set[TensorHandle] = set()
        self.outputs: Set[TensorHandle] = set()
        self.registry = registry

    def get_last(self, kind: Type, with_output: bool = True):
        if not with_output and len(self.ops[kind]) != 0:
            return self.ops[kind][-1]

        for op in reversed(self.ops[kind]):
            output = get_op_output(op)
            if output is not None:
                print("OP:", op)
                print("OUTPUT:", output)
                return output
        return None

    def tie_tensors(self):
        last_golden_op = self.get_last(Type.GOLDEN)
        last_device_op = self.get_last(Type.DEVICE)

        if last_golden_op is None and last_device_op is None:
            return

        self.registry.add_tensor(last_golden_op, last_device_op)

    def add_op(self, op: Op, kind: Type):
        self.ops[kind].append(op)


class Registry:
    def __init__(self):
        self.groups: dict[int, OpGroup] = {}
        self.tensors: Set[TensorHandle] = set()
        self.golden_map: dict[str, TensorHandle] = {}
        self.device_map: dict[str, TensorHandle] = {}

    def add_op(self, op: Op, kind: Type):
        key = location2key(op.location)
        if key not in self.groups:
            self.groups[key] = OpGroup(key, self)
        self.groups[key].add_op(op, kind)

    def tie_tensors(self):
        for group in self.groups.values():
            group.tie_tensors()

    def add_tensor(self, golden_value: Value | None, device_value: Value | None):
        tensor: TensorHandle = TensorHandle(golden_value, device_value)
        self.tensors.add(tensor)
        if golden_value is not None:
            self.golden_map[golden_value.get_name()] = tensor
        if device_value is not None:
            self.device_map[device_value.get_name()] = tensor


class GoldenEngine:
    pass


class RunSession:
    def __init__(self):
        self.registry = Registry()
        self.golden_module = None
        self.device_module = None
        self.golden_engine = None

    def _postop(self, binary, programContext, opContext):
        NotImplementedError()

    def _preop(self, binary, programContext, opContext):
        NotImplementedError()

    def register_hooks(self):
        callback_env_pre = DebugHooks.get(self._preop, self._postop)


def main(ttir_path, ttnn_path, main_fn):
    ir_ttir: RawIRModule = RawIRModule(ttir_path.read_text(), Context())
    ir_ttir.set_main_op(main_fn)
    ir_ttnn: RawIRModule = RawIRModule(ttnn_path.read_text(), Context())
    ir_ttnn.set_main_op(main_fn)

    def walk_ops(op):
        print(op)
        if hasattr(op, "arguments"):
            print("LOOOOL")
            pdb.set_trace()
        return WalkResult.ADVANCE

    ir_ttnn._main_op.operation.walk(walk_ops)

    registry: Registry = Registry()

    ttir: AnnotatedModule = AnnotatedModule(ir_ttir, registry, Type.GOLDEN)
    ttnn: AnnotatedModule = AnnotatedModule(ir_ttnn, registry, Type.DEVICE)

    ttir.register()
    ttnn.register()

    ttir_inputs = ir_ttir.get_inputs()
    ttnn_inputs = ir_ttnn.get_inputs()

    pdb.set_trace()

    # for ttir_input, ttnn_input in zip(ttir_inputs, ttnn_inputs):
    #     registry.add_tensor(ttir_input, ttnn_input)

    registry.tie_tensors()

    pdb.set_trace()


if __name__ == "__main__":
    folder = Path("/localdev/ndrakulic/chisel/mnist")
    ttir = folder / "ttir.mlir"
    ttnn = folder / "ttnn.mlir"
    main(ttir_path=ttir, ttnn_path=ttnn, main_fn="forward")
