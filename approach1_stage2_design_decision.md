# Stage 2 Design Decision — ADDITIVE `getResultTileStrides` (supersedes plan §2.1)

> **Status:** DECIDED & RATIONALE-RECORDED, supersedes the contract-change
> approach in `approach1_contract_phase1_plan.md` §2.1 (which proposed adding a
> `resultStrides` OUT-PARAM to the existing `getResultTilePosition`). Identified
> during Stage 2 execution (2026-06-29) after two grounded blast-radius findings.

## The flaw in plan §2.1 (verified, not theoretical)

The plan proposed adding a `resultStrides` out-param to `getResultTilePosition`,
claiming (§2.1): *"Default fills unit strides so every existing implementor is
unaffected (they never populate the new out-param; the default fires)."*

This claim is **FALSE**. The tablegen `defaultImplementation` only fires for ops
that do NOT override the method. Every existing implementor *overrides*
`getResultTilePosition`, so adding a param cascades to a forced signature update
on **all of them**. Ground truth (grep of the whole tree, 2026-06-29):

**9 MLIR implementors** (plan listed 4):
- `lib/Dialect/Linalg/Transforms/TilingInterfaceImpl.cpp:236` (linalg generic), `:1029` (Pack), `:1499` (Unpack)
- `lib/Dialect/Tensor/IR/TensorTilingInterfaceImpl.cpp:58` (tensor.pad)
- `lib/Dialect/Linalg/IR/LinalgOps.cpp:2976` (Softmax), `:3270` (WinogradFilter), `:3419` (WinogradInput), `:3599` (WinogradOutput)
- `test/lib/Dialect/Test/TestOpDefs.cpp:1247` (TilingNoDpsOp)

**~16 IREE implementors** (not in the mlir-opt build, but the END GOAL):
- `compiler/src/iree/compiler/Dialect/LinalgExt/IR/TilingInterfaceImpl.cpp` — 15 ops (Scatter/Gather/MapLoad/MapStore/Sort/Fft/Scan/Topk/TopkV2/ArgCompare/ExpReduction/Im2col/3×Winograd)
- `compiler/src/iree/compiler/Codegen/IR/TilingInterfaceImpl.cpp:144` (InnerTiledOp)

**Callers the plan missed entirely** (forced to consume a new out-param):
- `lib/Dialect/Linalg/Transforms/Split.cpp:51`
- `lib/Dialect/Linalg/Transforms/Tiling.cpp:750`
- internal self-calls: `LinalgOps.cpp:3328/3508/3679`, `TilingInterfaceImpl.cpp:1005/1264`
- IREE: `Codegen/Common/TileDispatchUsingInterface.cpp:253`, `PCF/Transforms/FuseConsumers.cpp:401`

A param change = editing ~25 implementors + ~8 callers before the fix can land.
High churn, high regression risk, and it does NOT achieve the plan's stated
"default carries unit" intent.

## The fix — ADDITIVE method (chosen)

Add a **new** interface method:

```tablegen
InterfaceMethod<
  /*desc=*/[{ ... the stride of each dim of the result tile (default unit) ... }],
  /*retType=*/"::llvm::LogicalResult",
  /*methodName=*/"getResultTileStrides",
  /*args=*/(ins "::mlir::OpBuilder &":$b, "unsigned":$resultNumber,
    "::mlir::ArrayRef<::mlir::OpFoldResult> ":$offsets,
    "::mlir::ArrayRef<::mlir::OpFoldResult> ":$sizes,
    "::mlir::SmallVector<::mlir::OpFoldResult> &":$resultStrides),
  /*defaultImplementation=*/[{ return ::mlir::failure(); }]
>
```

**Why this carries the plan's intent correctly:** it is a NEW method, so NONE of
the ~25 existing implementors override it → the default (`failure`) fires for all
of them. The SCF caller interprets `failure` as "unit strides" (sized from the
already-computed `resultSize`): `if (failed(op.getResultTileStrides(...)) || resultStride.empty()) resultStride.assign(resultSize.size(), b.getIndexAttr(1));`.
Only the Stage-3 `tensor.insert_slice` impl overrides `getResultTileStrides` to
return its non-unit `getMixedStrides()`. Identical capability; the stride still
flows impl → SCF helper → GenerateTiledBodyFn channel → writeback exactly as the
plan's §1.3 call chain specifies.

## Stage 2 edit set under the additive design (minimal)

| # | File:site | Change |
|---|---|---|
| 1 | `TilingInterface.td` (after `:162`) | add `getResultTileStrides` method, default `return failure()` |
| 2 | `TileUsingInterface.cpp:359` typedef `GenerateTiledBodyFn` | add `resultStrides` out-param |
| 3 | `TileUsingInterface.cpp:848` static helper | add `resultStride` out-param; FullReduction branch calls `op.getResultTileStrides` (unit fallback); PartialReduction branch fills unit |
| 4 | `TileUsingInterface.cpp:1158` body lambda | accept `resultStrides`; declare local `resultStride`, pass to helper, emplace into `resultStrides` |
| 5 | `TileUsingInterface.cpp` tiledBodyFn calls `:430/:607/:661/:701` | declare a `resultStrides` local and pass it |
| 6 | `TileUsingInterface.cpp` writeback `:447` (for), `:616` (forall) | read `resultStrides[i]` instead of local `getIndexAttr(1)` |

**UNTOUCHED (zero-touch, zero behavior change):** all 9 MLIR implementors, all
~16 IREE implementors, `Split.cpp`, `Tiling.cpp`, all internal self-calls, the
`.td` `declareMethods` lists, `YieldTiledValuesFn` (fusion path `:951/:1006`),
`GenerateLoopTerminatorFn` (custom-loop path keeps unit — probe uses forall),
`PartialReductionOpInterface` (the similarly-named method at `TilingInterface.td:485`,
a different interface — do NOT touch).

## Why this still satisfies the GO gate & call chain

The probe's call chain (plan §1.3) is unchanged in TOPOLOGY:
`tile_using_forall` → `tileToForallOpImpl` → `tileUsingSCF` → body lambda →
`getResultTilePosition` (helper) → now ALSO `getResultTileStrides` (helper) →
`:616-617` writeback. The stride enters at the `getResultTileStrides` hop instead
of as an out-param of `getResultTilePosition`, but reaches `:616-617` through the
same `GenerateTiledBodyFn` channel. G3 (strided writeback for insert_slice),
G4a/G4b (EXEC) are all achievable. R1 (propagate-first/flip-last) holds: the
channel is unit everywhere until the Stage-3 impl overrides `getResultTileStrides`.

## Risks / honest limits

- The fusion path (`YieldTiledValuesFn`) is NOT threaded. It keeps hardcoded unit.
  Correct for all in-scope ops (insert_slice fusion is out of scope — the Stage-3
  impl deliberately omits operand-tile methods). If a future strided fused op is
  needed, thread `getResultTileStrides` there too.
- The custom-loop path (`generateLoopNestUsingCustomOp`) keeps unit strides (its
  `GenerateLoopTerminatorFn` is unchanged). Correct — the probe uses forall.
- This is a LOCAL vendored patch. Upstreaming an additive interface method is a
  milder review ask than a signature change to a core method (favorable).
