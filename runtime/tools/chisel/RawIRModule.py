# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

"""Light‑weight, read‑only wrapper around an MLIR module text.

The class is intentionally minimal: it only parses the textual IR once,
exposes a couple of convenience walks and lets higher‑level components
('OpManager', 'ChiselContext', …) do the heavy lifting.

All mutating helpers (save / clone / re‑write) are *deliberately* left out –
`RawIRModule` is treated as an immutable snapshot.
"""

from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Sequence

import ttmlir  # type: ignore
from ttmlir.ir import Context, Module, Operation  # type: ignore

__all__ = [
    "RawIRModule",
]


class RawIRModule:
    """A parsed MLIR module plus some convenience helpers.

    Parameters
    ----------
    text
        Entire textual MLIR (e.g. from ``Path.read_text()``).
    kind
        Whether the module is the golden reference or device‑lowered version.
    ctx
        An existing :pyclass:`ttmlir.ir.Context` to parse into.  If *None*, a
        private Context that has all available dialects pre‑loaded is created.
    """

    class Kind(str, Enum):
        GOLDEN = "golden"
        DEVICE = "device"

        # Keep Enum values lowercase strings so that ``str(kind)`` is usable in
        # filenames / logs without additional mapping.

    # ---------------------------------------------------------------------
    # Construction helpers
    # ---------------------------------------------------------------------

    def __init__(
        self,
        text: str,
        kind: "RawIRModule.Kind",
        ctx: Context,
    ) -> None:
        self.kind: RawIRModule.Kind = kind

        # Parse once – downstream code only deals with Operation handles.
        self.context: Context = ctx
        with self.context:
            self._module: Module = Module.parse(text)

        # Freeze the canonical textual form for easy caching / diff‑ing.
        self.text: str = self._module.operation.get_asm()

        # Resolve the top‑level user‑visible entry once (expensive recursive
        # search).
        self._main_func: Operation | None = None

    @property
    def mlir_module(self) -> Module:
        """The parsed :pyclass:`ttmlir.ir.Module` (read‑only)."""

        return self._module

    # The *device* IR sometimes nests the real payload function under a couple
    # of wrappers (\"tt.device\", inner :mlir:`module`, …).  A generic DFS is
    # safer than trying to hard‑code every variant we may meet.
    def get_main_function(self, *, name: str) -> Operation:
        """Return the first :pyclass:`ttmlir.ir.Operation` representing a
        ``func.func`` with the given *symbol* ``name``.

        If *name* is *None*, the heuristic is:

        1. Any function tagged with ``@tt.entry_func`` or ``sym_visibility`` =
           "public" wins.
        2. Otherwise, the first encountered ``func.func`` is returned.
        """

        if self._main_func is not None and name is None:
            return self._main_func

        def _dfs(op: Operation) -> Optional[Operation]:
            # Check *this* op.
            if op.name == "func.func":
                if name is None:
                    return op
                sym = op.attributes.get("sym_name")
                if sym is not None and str(sym) == name:
                    return op
            # Recurse into regions → blocks → operations.
            for region in op.regions:
                for block in region.blocks:
                    for inner in block.operations:
                        hit = _dfs(inner)
                        if hit is not None:
                            return hit
            return None

        hit = _dfs(self._module.operation)
        if hit is None:
            target = f'"{name}"' if name else "entry"
            raise ValueError(f"Could not find main function {target} in module")

        if name is None:
            self._main_func = hit  # cache
        return hit

    # Two flavours of *walk* as requested.  Both accept a Python *callable* that
    # takes the current Operation and (optionally) returns ``bool`` whose *False*
    # value stops the traversal early (mirrors :py:meth:`Operation.walk`).

    def walk_all_ops(self, callback: Callable[[Operation], bool | None]) -> None:
        """Depth‑first walk over *all* operations (including nested
        functions)."""

        _ = self._module.operation.walk(callback)

    def walk_main_body_ops(
        self,
        callback: Callable[[Operation], bool | None],
        *,
        recursive: bool = False,
    ) -> None:
        """Walk only the *body* of the main function.

        Parameters
        ----------
        callback
            Called for every operation encountered.
        recursive
            If *False* (default) the traversal is *flat* – nested regions inside
            the body are **not** entered.  If *True*, the walk behaves like
            :py:meth:`walk_all_ops` but is limited to the entry function.
        """

        main_func = self.get_main_function()
        body_block = main_func.regions[0].blocks[0]

        def _flat_walk(block_ops: Sequence[Operation]) -> None:
            for op in block_ops:
                res = callback(op)
                if res is False:
                    break

        if recursive:
            for op in body_block.operations:
                # A naive recursive walk over the *block* so we do not enter
                # sibling functions elsewhere.
                op.walk(callback)
        else:
            _flat_walk(body_block.operations)
