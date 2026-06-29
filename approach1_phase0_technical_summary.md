# Approach 1 — Phase 0 Technical Summary (handoff for IREE-side planning)

> **Purpose:** a precise, grounded record of what the MLIR-side work delivered,
> what it proves, and what remains — written for the *next* planning phase
> (IREE-side integration). Phase 0 is COMPLETE and GO-gated. Generated
> 2026-06-29.
>
> **Patch:** `~/Developer/iree/mlir-strided-insert-slice-tiling.patch`
> (8 files, applies cleanly to a fresh checkout of `third_party/llvm-project`
> at HEAD `22da7f929139`). Design-decision trail:
> `approach1_stage2_design_decision.md`; superseded plan:
> `approach1_contract_phase1_plan.md` (§2.1 design was pivoted — see below).

---

## 1. The problem & the capability now proven

**Original goal:** IREE failed to compile `m[0::2,0::2]=True` (a doubly-strided
scatter-fill) on `llvm-cpu`/`metal-spirv`. Root cause: a transposed stride
mismatch (fill region `[1,2]` vs output store `[2,1]`) forced one-shot
bufferization to emit a bare strided store that the tiler couldn't express.

**Phase 0 question (binary GO):** can MLIR's SCF tiler, via the
`TilingInterface` contract, emit a **non-unit-strided per-tile writeback** for a
real `tensor.insert_slice`? Proven end-to-end by execution.

**Answer: YES.** The bidirectional case that motivated the whole effort now
works: dilating a `<2x2>` source over a `<4x4>` dest with stride `[2,2]`
(every other row AND column) produces the exact checkerboard
`[[1,0,1,0],[0,0,0,0],[1,0,1,0],[0,0,0,0]]`, lowered and executed via
`mlir-runner`.

**What GO does NOT claim:** `m[0::2,0::2]=True` does NOT yet compile in IREE.
That needs dynamic `tensor<?x?xi8>`, IREE anchor selection + filter widening,
and end-to-end IREE lowering survival — all deferred (§7).

---

## 2. The mechanism (design — read this before extending)

### 2.1 Additive contract method, NOT a signature change
A new `TilingInterface` method `getResultTileStrides` was added with a
`return failure()` default. **Critically, this is additive, not a param on
`getResultTilePosition`.** Reason (verified): all ~25 implementors of
`getResultTilePosition` (9 in MLIR: linalg/Pack/Unpack/pad/Softmax/3×Winograd/
TilingNoDpsOp; ~16 in IREE incl. LinalgExt's 15) *override* it, so adding an
out-param would cascade to forced updates on all of them. A new method's
default *does* fire for every non-overriding op → unit strides everywhere,
zero-touch to existing implementors. Only `tensor.insert_slice` overrides it.
(Lesson captured as managed skill `extend-mlir-opinterface-additively`.)

### 2.2 The stride flow (verified call chain)
```
tile_using_forall(InsertSliceOp)
  → tileUsingSCF → body lambda (GenerateTiledBodyFn)
      → static helper getResultTilePosition(..., resultStride&)   [SCF]
          FullReduction branch:
            op.getResultTilePosition(...)      → resultOffsets/resultSizes
            op.getResultTileStrides(...)       → resultStrides   ← THE HOP
              (default failure → helper fills unit, sized by resultSize)
          PartialReduction branch: fills unit
      → resultStrides channeled through GenerateTiledBodyFn out-param
  → generateLoopNestUsingForallOp writeback (:616)
      tensor.parallel_insert_slice %tile into %o[off][size][resultStrides]
```
The `resultStrides` channel lives on `GenerateTiledBodyFn` (typedef +
static helper + body lambda + 4 `tiledBodyFn` call sites + 2 writeback sites
ForOp/ForallOp). The fusion path (`YieldTiledValuesFn`) and custom-loop
terminator are **intentionally untouched** — they keep hardcoded unit strides
(correct for all in-scope ops).

### 2.3 The `InsertSliceOpTiling` ExternalModel (the strided anchor)
`getIterationDomain`  = the **source** tensor shape (`tensor::getMixedSizes`).
`getTiledImplementation` = contiguous `tensor.extract_slice %source[off][size][1...]`.
`getResultTilePosition`  = `resultOffset[d] = base[d] + iterOffset[d]*stride[d]`
                            (affine map `d0 + d1*s0`); `resultSizes = iterSizes`
                            (the source/`%tile` shape — `verifyInsertSliceOp` is
                            sizes-only, so the dest span is implicit).
`getResultTileStrides`   = `getMixedStrides()`.
Bails on **rank-reduced** insert_slice (source rank ≠ dest rank). Deliberately
**omits** the operand-tile/fusion methods → fusion consumers get default
`failure()` (containment by construction for the fusion direction).

---

## 3. Code changes (3 files)

| File | Change |
|---|---|
| `include/mlir/Interfaces/TilingInterface.td` | New `getResultTileStrides` `InterfaceMethod` (default `return failure()`), placed after `getResultTilePosition`. |
| `lib/Dialect/SCF/Transforms/TileUsingInterface.cpp` | `GenerateTiledBodyFn` typedef gains `resultStrides&`; static helper `getResultTilePosition` gains `resultStride&` (fetches via `getResultTileStrides`, unit fallback; PartialReduction fills unit); body lambda populates it; 4 `tiledBodyFn` call sites pass it; ForOp & ForallOp writebacks read `resultStrides[i]` instead of hardcoded `getIndexAttr(1)`. |
| `lib/Dialect/Tensor/IR/TensorTilingInterfaceImpl.cpp` | New `InsertSliceOpTiling` ExternalModel (modeled on `PadOpTiling`); registered next to `PadOp` in `registerTilingInterfaceExternalModels`. |

Exact diffs: see the patch. Cite by symbol name (lines shift); key symbols:
`getResultTileStrides` (.td), `GenerateTiledBodyFn` / `getResultTilePosition`
static helper / `generateLoopNestUsingForallOp` (TileUsingInterface.cpp),
`InsertSliceOpTiling` / `registerTilingInterfaceExternalModels`
(TensorTilingInterfaceImpl.cpp).

---

## 4. Tests (5 files)

| File | Kind | Proves |
|---|---|---|
| `test/Dialect/Tensor/tiling-insert-slice-strided.mlir` | IR emission (no canon) | sub-case A genuine-scatter writeback `[2,4][2,1]`; sub-case B offset; 2 R4 unit-stride twins. **G3.** |
| `test/Dialect/Tensor/tiling-insert-slice-offset.mlir` | IR, `-canonicalize` | Per-piece offset localization: off_caseA literal `[0,0]`, off_caseC non-zero base `[2,0]` (pins base-addition), off_caseB `affine.apply` stride-mult. |
| `test/Integration/Dialect/Tensor/CPU/strided-insert-slice.mlir` | EXEC `mlir-runner` | **G4a** tile_sizes: rows {0,2}=1, full-cell. |
| `.../strided-insert-slice-num-threads.mlir` | EXEC | **G4b** num_threads `[2,1]` multi-tile: rows {0,2,4,6}=1 (real-IREE offset path). |
| `.../strided-insert-slice-dilate2x2.mlir` | EXEC | **Bidirectional `[2,2]`** (the original goal): checkerboard, exercises dim-1 striding. |

---

## 5. Verification (grounded, not asserted)

- **Build:** `MLIR_INCLUDE_TESTS=ON`; `mlir-opt` + `mlir-runner` + runner-utils
  build green.
- **G3 (IR):** RED→GREEN. Initially RED at `LinalgTransformOps.cpp:3899`
  (`dyn_cast<TilingInterface>` rejected insert_slice); after Stage 3 the
  strided writeback emits. Sub-case A is load-bearing: a hardcoded `[1,1]`
  writeback cannot match `[2,4][2,1]` (not the vestigial-stride trap).
- **G4a/G4b EXEC:** `mlir-runner` exit 0; matrices verified full-cell.
  A buggy contiguous `[1,1]` writeback would give rows {0,1}=1; actual {0,2}=1
  proves the stride scattered.
- **Mutation-proven (localization):**
  - *Offset:* dropped the base (`d0+d1*s0`→`d1*s0`) → off_caseC failed with
    actual `[0,0]` vs expected `[2,0]`, sizes/strides still matched → localized
    to offset/base.
  - *EXEC numerics:* dropped stride propagation (`getResultTileStrides`→
    `failure()`) → runner produced rows {0,1}=1 → full-cell check failed →
    proves the EXEC test catches wrong values end-to-end.
- **Zero regression:** `transform-op-fuse.mlir` + matmul/generic/pad tiling
  probes all green after the contract change and the new TilingInterface
  membership.

---

## 6. Build & run environment (required for EXEC)

- `MLIR_INCLUDE_TESTS=ON` is **mandatory** for EXEC: it provides
  `-test-transform-dialect-erase-schedule` (without it the transform schedule
  survives and `mlir-runner` fails — it `translateModuleToLLVMIR`s the *whole*
  module) and `--test-lower-to-llvm` (bundles the strided-memref lowering:
  `expand-strided-metadata` + `finalize-memref-to-llvm`, proven to lower strided
  `memref.copy`).
- Build targets are **lowercase**: `mlir-opt`, `mlir-runner`, `mlir_c_runner_utils`,
  `mlir_runner_utils` (NOT `MLIRCRunnerUtils`; `mlir-cpu-runner` isn't a target
  here — use `mlir-runner`).
- EXEC pipeline (mmt4d template):
  `mlir-opt %s -transform-interpreter -test-transform-dialect-erase-schedule
  -one-shot-bufferize="bufferize-function-boundaries" -buffer-deallocation-pipeline
  -cse -canonicalize --test-lower-to-llvm -o %t` then
  `mlir-runner %t -e <entry> -entry-point-result=void
  -shared-libs=libmlir_runner_utils.dylib,libmlir_c_runner_utils.dylib`.
  (`-one-shot-bufferize` option is `bufferize-function-boundaries`; the plan's
  `allow-return-allocs` does not exist in this version.)
- Captured as managed skill `run-vendored-mlir-transform-tests`.

---

## 7. Scope boundaries (what is NOT proven — load-bearing for next planning)

1. **Dynamic `tensor<?x?xi8>`** — the real IREE dispatch shape. The probe is
   static; stride-dividing a dynamic span needs runtime `ceildivi` + a dynamic
   iteration domain. **Unproven.**
2. **IREE anchor selection + `computeOp` filter widening** — `insert_slice` is
   now `TilingInterface`, but IREE's `isComputeOp` / lowering-config anchor
   selection (`TileDispatchUsingForall.cpp`) must be taught to *select* it.
3. **§4a containment is NOT implemented** (see §8) — the impl is **UNGATED**.
4. **Rank-reduced insert_slice** — bails (`getSourceType().getRank() !=
   getDestType().getRank()`); rule stated but unimplemented.
5. **Fusion of strided slices** — the `YieldTiledValuesFn` path is untouched
   (unit strides). The impl omits operand-tile methods.
6. Arbitrary/coprime strides, dim-1 transpose `[1,2]` lit — only static `[2,1]`
   and `[2,2]` are tested.
7. `LLVM #51660` (vector strided load/store) — not on the compile path; strided
   `memref.copy` lowers to a correct scalar copy (slow, not wrong).

---

## 8. Next-stage (IREE-side) change sites & open questions

The MLIR capability is proven; landing it in IREE needs (all unimplemented):

- **Containment FIRST (§4a — the impl is ungated).** Every bare
  `dyn_cast<TilingInterface>` walk now matches `insert_slice`. Known dangerous
  sites in the real bug's pipeline:
  - `compiler/src/iree/compiler/Codegen/Common/GPU/GPUGreedilyDistributeToThreads.cpp:139`
    (`dyn_cast<TilingInterface>` in an IR walk → routes insert_slice to
    `tileToThreads`).
  - `compiler/src/iree/compiler/Codegen/Common/TileAndFuseUtils.cpp`
    (`:40, :78, :90, :271, :394` fusion worklist casts).
  - **Two firewalls (from the plan):** (1) marker-gate on the impl's
    `getTiledImplementation`/`getResultTilePosition` (return failure unless
    marked) — best-effort; (2) IREE anchor allow-list (the HARD firewall —
    filter on an explicit allow-list, not bare `TilingInterface`, before
    admitting insert_slice as distributable/fusable). **The allow-list MUST
    ship before the ungated impl lands in any IREE pipeline.**
- **Anchor selection:** `TileDispatchUsingForall.cpp:67-76` (anchor = last
  `computeOp` with a workgroup lowering config). Make `insert_slice` selectable
  as a (co-)anchor so the strided write lives in the forall body directly.
- **Filter widening:** `TileAndFuseUtils.cpp:141` (`dyn_cast<ParallelInsertSliceOp>`
  seed) and `:154-155` (`filterFn`) — relax to admit a strided outside-forall
  `insert_slice` consumer. (Note: the feasibility doc `approach1_tensor_level_fusion.md`
  §4.3 has proposals here, but they're from the **REFUTED consumer-fusion
  approach** — re-derive for the contract-based initial-tiling mechanism, do
  not execute as-is.)
- **Source-load co-distribution:** the strided write's source (`flow.dispatch.tensor.load`
  → `tensor.insert_slice`) path through `BufferizeDispatchTensorLoadStore.cpp`.
- **Verify:** `verifyComputeOpsAfterDistribution` (`TileDispatchUsingForall.cpp:196-209`)
  still holds once the strided insert is in the dispatch.
- **Dynamic iteration domain:** `getIterationDomain` uses `tensor::getMixedSizes`
  (already dynamic-friendly), but the stride-divided *iteration domain* under
  dynamic sizes is unproven — needs a dedicated test.

**Open question for the planner:** is the IREE-side fix better attacked as
(a) making `insert_slice` a dispatch anchor (initial tiling, matching the MLIR
mechanism proven here), or (b) consumer-fusion of the strided slice into a
tiled fill (the refuted approach, possibly viable now that the slice has a
TilingInterface impl)? The MLIR work supports (a) directly; (b) would need the
fusion path threaded (currently untouched).

---

## 9. Artifacts index

- **Patch:** `~/Developer/iree/mlir-strided-insert-slice-tiling.patch`
- **Design decision:** `~/Developer/iree/approach1_stage2_design_decision.md`
- **Plan (superseded §2.1, rest still useful context):** `approach1_contract_phase1_plan.md`
- **Feasibility doc (IREE-side detail, §4.3 is REFUTED approach):** `approach1_tensor_level_fusion.md`
- **Memory:** Phase-0 status + bidirectional-case-works retained in Mnemopi.
- **Skills:** `run-vendored-mlir-transform-tests`, `extend-mlir-opinterface-additively`.

---

## 10. Check-in & CI notes

- **The 2 IR-emission tests** (`test/Dialect/Tensor/tiling-insert-slice-strided.mlir`,
  `tiling-insert-slice-offset.mlir`) run in **any** build — plain
  `mlir-opt --transform-interpreter [--split-input-file [-canonicalize]] | FileCheck`.
- **The 3 EXEC tests** (`test/Integration/Dialect/Tensor/CPU/strided-insert-slice*.mlir`)
  require **`MLIR_INCLUDE_TESTS=ON`**. They use `-test-lower-to-llvm`,
  `-test-transform-dialect-erase-schedule`, and the `printMemrefI32` runner util
  (`libmlir_c_runner_utils`), none of which register in a tests-off build.
- **Implication:** fine for upstream MLIR (tests-on is standard there). In
  IREE's vendored config (`MLIR_INCLUDE_TESTS=OFF` by default), those 3 EXEC
  tests will not run / will fail to compile-run unless tests are enabled.
  Before landing in IREE CI, either enable `MLIR_INCLUDE_TESTS=ON` in the
  relevant build or expect those 3 to be skipped. The 3 source files
  (`TilingInterface.td`, `TileUsingInterface.cpp`, `TensorTilingInterfaceImpl.cpp`)
  are unaffected by the test-config toggle — only EXEC-test *execution* depends
  on it.
