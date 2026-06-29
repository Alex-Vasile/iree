# Approach 1 (NOTE §4-1) — Fuse the strided output store at the TENSOR level

**Scope:** READ-ONLY feasibility investigation. Every claim is grounded in a
file:line the author personally opened. Reasoned (not directly observed)
claims are marked `[INFERENCE]`.

**Subject dispatch:** `m[0::2, 0::2] = True` — a doubly-strided, dynamic,
out-of-place scatter-fill that fails to compile on both `llvm-cpu` and
`metal-spirv`. Failing per-pass IR: `dump.mlir:31648-31679` (llvm-cpu).

---

## 1. Verdict (one paragraph)

The approach is **sound in direction but blocked by a structural, upstream
limitation, not by IREE's filter.** The IREE-side consumer-fusion gate
(`fuseConsumersIntoForall`) can be widened trivially, but doing so only
exposes the real wall: **upstream MLIR's structured-ops stack cannot represent
a strided tensor write-back at all.** The root cause is *not* an SCF choice but
the `TilingInterface` contract itself
(`mlir/include/mlir/Interfaces/TilingInterface.td`): none of its five core
methods — `getTiledImplementation`, `getResultTilePosition`,
`generateResultTileValue`, `getTiledImplementationFromOperandTiles`,
`getIterationDomainTileFromOperandTiles` — accepts or returns strides; a tile is
modelled as a pure affine-identity `(offset, size)` region (see §2.4.1). As a
*consequence*, SCF's `TileUsingInterface.cpp` hardcodes unit strides in **six**
write-back/dest-slice construction sites (`:447-448`, `:616-617`, `:951-955`,
`:1006-1007`, `:1565-1566`, `:2366-2367`) and **explicitly rejects**
non-unit-stride candidates in two (`:2313-2317` consumer, `:1502-1504`
producer), all via the shared `isOneInteger` helper (`StaticValueUtils.h:38`).
The same unit-stride assumption recurs across linalg and tensor transforms
(§8.2) and is corroborated — still open since 2021 — at the *vectorization*
layer by LLVM issue #51660 (§8.1).
Because of this, the existing precedent that *does* compile — the
single-stride control `m[:,0::2]` — compiles **not because fusion handled a
stride, but because flow-formation + one-shot bufferization were able to
represent the whole write as a contiguous in-place alias** (matching
`[1,1]`/`[1,1]` region and output). The failing two-axis case has a
**transposed** stride mismatch — region `[1,2]` (dim-1 strided) vs output
`[2,1]` (dim-0 strided) — which bufferization cannot alias, so a bare copy
of a `#hal.descriptor_type<storage_buffer>` is emitted outside the workgroup
forall and the verifier correctly rejects it. **Independent effort rating:
HARD+, greater than the "3d+" the expert review gave.** The dominant cost is
*either* patching upstream `mlir/lib/Dialect/SCF/Transforms/TileUsingInterface.cpp`
(a foundational file used by every SCF-tiling dialect, very high blast radius
and an upstream-acceptance battle) *or* building an IREE-local, stride-aware
tiling path that bypasses upstream's unit-stride assumption. The fix is the
principled end state, but it is the wrong first move; it should be gated on
the cheaper alternatives.

---

## 2. Mechanism today (what the tiler + fusion does)

### 2.1 Anchor selection — the fill is the only anchor

`TileAndDistributeToWorkgroupsUsingForallOpPass::runOnOperation` calls
`getTiledAndDistributionInfo(rewriter, computeOps)` (`TileDispatchUsingForall.cpp:61-142`,
invoked at `:234-235`). The anchor is selected at `:67-76`:

```cpp
Operation *tilableOp = nullptr;
for (Operation *op : llvm::reverse(computeOps)) {
  if (getLoweringConfig(op)) {
    if (!getLoweringConfig(op).hasWorkgroupTilingLevel()) { continue; }
    tilableOp = op;
    break;
  }
}
```

It picks the **last** `computeOp` (reverse walk) that carries a lowering
config with a workgroup tiling level. `isComputeOp` is defined at
`Utils.cpp:980-982`:

```cpp
bool isComputeOp(Operation *op) {
  return isa<TilingInterface, IREE::Codegen::UKernelOpInterface>(op);
}
```

So `computeOps` = all `TilingInterface`/`UKernel` ops. In the failing
dispatch the anchor is the inner `linalg.generic { yield %c1_i8 }` fill
(`dump.mlir:31665`). The strided `tensor.insert_slice %36 into %35[…][1,2]`
(`dump.mlir:31678`) is **not** a `TilingInterface` op — the tensor dialect
registers `TilingInterface` only for `tensor::PadOp`
(`lib/Dialect/Tensor/IR/TensorTilingInterfaceImpl.cpp:314`), never for
`insert_slice`/`extract_slice`. It is therefore **not a `computeOp`** at all
(`isComputeOp` requires `TilingInterface | UKernelOpInterface`,
`Utils.cpp:980-982`), so it is never collected into `computeOps`, never
selectable as the anchor, and never tiled. *(Correction 2026-06-29: an earlier
version of this section claimed `insert_slice` "is a `TilingInterface` op and
therefore a `computeOp`, but has no lowering config" — that is **false**; the
no-lowering-config point is moot. This error propagated into the Phase-1 plan's
SCF-only consumer-fusion design, which is refuted by the same fact plus an
architectural one: consumer fusion reads only the loop-**internal** unit-stride
candidate (`getProducingParallelInsertSlice`, `TileUsingInterface.cpp:2487`),
never the external strided store, so it can preserve but not create a stride —
the contract change is a prerequisite, not a fallback. See
`expert_review_phase1_plan.md` §3.)*

### 2.2 Tiling produces the forall; the fill's writeback is unit-strided

`runOnOperation` tiles the anchor with
`scf::tileConsumerAndFuseProducersUsingSCF` (`TileDispatchUsingForall.cpp:353-355`).
That produces the workgroup `scf.forall` whose body holds the tiled fill and a
`tensor.parallel_insert_slice` writeback into `shared_outs`
(`dump.mlir:31652-31677`). Critically, the **writeback the tiler itself emits
always has unit strides** — see §2.4. The `shared_outs` init here is the
**dim-1-strided** region `%extracted_slice = tensor.extract_slice %35[…][1,2]`
(`dump.mlir:31651`).

### 2.3 The consumer-fusion filter — two gates, neither accepts the strided store

After producer tiling, `runOnOperation` calls `fuseConsumersIntoForall`
(`TileDispatchUsingForall.cpp:370-375`):

```cpp
FailureOr<std::queue<Operation *>> newFusionOpportunities =
    fuseConsumersIntoForall(
        rewriter, tileAndFuseResult->tiledAndFusedOps.getArrayRef(),
        tilingLoops, [&tiledAndFusedOps](Operation *op) {
          return tiledAndFusedOps.contains(op);
        });
```

Inside `fuseConsumersIntoForall` (`TileAndFuseUtils.cpp:112-247`), the
`addCandidateSlices` lambda has **two sequential gates**:

1. **`dyn_cast<tensor::ParallelInsertSliceOp>` at `TileAndFuseUtils.cpp:141`.**
   For each user of a tiled op's result, only a
   `tensor::ParallelInsertSliceOp` is recognized as the seed; everything else
   (`continue`). This is the gate the NOTE identifies.

2. **The `filterFn` at `TileAndFuseUtils.cpp:154-155`** (`filter_range(loopResult.getUsers(), filterFn)`).
   The filter passed from `runOnOperation` is `tiledAndFusedOps.contains(op)`.
   So a *consumer* of the forall result (e.g. the outside-forall
   `tensor.insert_slice %36 into %35`) is kept **only if it is already in the
   tiled-and-fused set** — which consumers never are. This is a second,
   independent strand.

Net effect: the outside-forall `tensor.insert_slice %36 into %35[…][1,2]`
(`dump.mlir:31678`) and the `iree_codegen.store_to_buffer` (`dump.mlir:31679`)
are **never** made fusion candidates. The forall ends with only the fill
inside; the strided write-back is stranded at dispatch scope.

Each accepted candidate is then fused via
`mlir::scf::tileAndFuseConsumer(rewriter, entry.fusableUser, loops)`
(`TileAndFuseUtils.cpp:215-216`).

### 2.4 What upstream `tileAndFuseConsumer` actually does (and why strides die)

`mlir::scf::tileAndFuseConsumer` lives at upstream
`mlir/lib/Dialect/SCF/Transforms/TileUsingInterface.cpp:2521-2570`. It
requires the consumer to implement `TilingInterface` (`:2524-2527` — else
"unhandled consumer"), collects the operands that come from the loop
(`:2540-2545`), finds the **producing insert-slice-like op** for each
(`getProducingInsertSliceLikeOp`, `:2479-2519`; for a `scf.forall` this is the
`tensor.parallel_insert_slice`, `:2484-2487`), and dispatches to
`tileAndFuseConsumerOfSlicesImpl` (`:2568-2569`).

`tileAndFuseConsumerOfSlicesImpl` (`:2205-2416`) is where strides are lost.
A grep across the file for `getMixedStrides` / `isOneInteger` / `getIndexAttr(1)`
confirms **six hardcoded-unit-stride construction sites plus two explicit
rejections**, all via the shared helper `isOneInteger` (`StaticValueUtils.h:38`,
defined as `isConstantIntValue(v, 1)`).

**A. Two explicit correctness rejections** (the gates a candidate hits first):

- **Consumer-fusion candidate rejection — `:2313-2317`:**
  ```cpp
  // 9. Check all insert stride is 1.
  if (!llvm::all_of(strides, isOneInteger)) {
    return rewriter.notifyMatchFailure(
        candidateSliceOp, "containingOp's result yield with stride");
  }
  ```
  `strides` come from `candidateSliceOp.getMixedStrides()` (`:2311`).

- **Producer-fusion sibling rejection — `:1502-1504`:**
  ```cpp
  // expect all strides of sliceOp being 1
  if (!llvm::all_of(sliceOp.getMixedStrides(), isOneInteger))
    return failure();
  ```

**B. Six write-back / dest-slice construction sites** that hardcode
`rewriter.getIndexAttr(1)` (so even if a candidate passed the rejections, its
strides would be silently dropped → wrong memory locations). They come in three
paired families — *initial tiling* vs *fusion add-init*, and *consumer* vs
*producer* dest:

- **Forall writeback — `:616-617` (initial tiling, `generateLoopNestUsingForallOp`) and `:1006-1007` (fusion add-init, `yieldTiledValuesAndReplaceLoop<scf::ForallOp>`):**
  ```cpp
  SmallVector<OpFoldResult> resultStride(resultOffset.size(),
                                         rewriter.getIndexAttr(1));
  tensor::ParallelInsertSliceOp::create(rewriter, loc, tiledValue, iterArg,
                                        resultOffset, resultSize, resultStride);
  ```
  *(The first pass of this doc listed only `:1006-1010`; `:616-617` is the
  parallel initial-tiling site and was missed.)*

- **`scf.for` writeback — `:447-448` (initial tiling, `generateLoopNestUsingForOp`) and `:951-955` (fusion add-init, `yieldTiledValuesAndReplaceLoop<scf::ForOp>`):** the same hardcoded-`1` `InsertSliceOp` emission. *(`:951-955` was missed.)*

- **DPS dest `extract_slice` — `:2366-2367` (consumer fusion, `tileAndFuseConsumerOfSlicesImpl`) and `:1565-1566` (producer fusion, `yieldReplacementForFusedProducer`):**
  ```cpp
  SmallVector<OpFoldResult>(resultOffsets[index].size(),
                            rewriter.getIndexAttr(1))
  ```
  *(The first pass listed only `:2363-2367`; `:1565-1566` is the producer-fusion
  parallel and was missed.)*

### 2.4.1 Root cause — one layer up: the `TilingInterface` contract has no strides

The six SCF hardcodes are not an arbitrary SCF decision; they are the *only
consistent value* given the contract one layer up. The `TilingInterface`
(`mlir/include/mlir/Interfaces/TilingInterface.td`) exposes five core methods,
and **none of them accepts or returns strides**:

| Method | Stride parameter? |
|---|---|
| `getTiledImplementation(offsets, sizes)` | none |
| `getResultTilePosition(offsets, sizes) → (resultOffsets, resultSizes)` | none in, none out |
| `generateResultTileValue(resultNumber, offsets, sizes)` | none |
| `getTiledImplementationFromOperandTiles(operandNumbers, allOffsets, allSizes)` | none |
| `getIterationDomainTileFromOperandTiles(operandNumbers, allOffsets, allSizes) → (iterOffsets, iterSizes)` | none |

The interface models a tile as a pure **affine-identity** region: iteration-domain
tile coordinate maps 1:1 to result/memory position. SCF *cannot* emit a strided
writeback because it has **no stride source** — `getResultTilePosition` returns
only offsets/sizes, and the implementors drop strides on the floor. Confirmed in
the linalg implementor (`Linalg/Transforms/TilingInterfaceImpl.cpp:236-258`):
`getResultTilePosition` builds `resultOffsets`/`resultSizes` from a slice-param
helper that returns only `.offsets`/`.sizes` — strides are never computed.

The smoking-gun comment is upstream at
`Tensor/Transforms/SwapExtractSliceWithProducerPatterns.cpp:31`:
```cpp
// `TilingInterface` currently only supports strides being 1.
if (!llvm::all_of(sliceOp.getMixedStrides(), isOneInteger))
  return failure();
```
"currently" = by interface contract, not by SCF accident. **This is the most
important finding of this approach: a strided `[1,2]` or transposed `[2,1]`
write-back is unrepresentable not because SCF chose to reject it, but because the
`TilingInterface` gives the tiler no stride to carry.** The two rejections at
`:2313-2317` / `:1502-1504` are therefore correctness guards *forced* by the
contract — removing them without adding strides to the interface (and its five
implementors) would silently miscompile.

*Git provenance:* the consumer-fusion API and the two operand-tile interface
methods landed in PR #88712 (Abhishek Varma, 2024-06-01); the consumer
multi-operand refactor that owns the current `:2313-2317` line is PR #145193
(MaheshRavishankar, 2025-06-25); the producer `:1502-1504` check is PR #93144
(Yun-Fly, 2024-06-28).

### 2.5 `store_to_buffer` is a non-`TilingInterface` op

`iree_codegen.store_to_buffer` (`StoreToBufferOp`) is defined at
`IREECodegenOps.td:230-250` with trait list
`[DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]` only — **no
`TilingInterface`.** (The `TilingInterface` declaration at `:264` belongs to
`InnerTiledOp`, `:256`, an unrelated op.) It is created by replacing
`iree_tensor_ext.dispatch.tensor.store` in
`BufferizeDispatchTensorLoadStore.cpp:70-88`. Because it is not
`TilingInterface`, it is neither a `computeOp` (`Utils.cpp:980-982`) nor an
acceptable `tileAndFuseConsumer` consumer (`TileUsingInterface.cpp:2524-2527`).
So the stranding can in principle be attacked *before* `store_to_buffer` is
formed — at the `flow.dispatch.tensor.store`/`tensor.insert_slice` stage —
which is exactly what §3 generalizes.

---

## 3. The single-stride precedent (the concrete template to generalize)

*(Per the main-agent amendment: the closest existing precedent is the
single-stride control `m[:,0::2]`, which compiles on llvm-cpu. This section is
the design template; §4 generalizes it.)*

### 3.1 What the control looks like, pre-bufferize

`control_dump.mlir:25781-25809` (the dispatch function the instant before
`IREEComprehensiveBufferizePass`):

```mlir
%subview = memref.subview %24[0, 0] [%22, %25] [1, 1]
    : memref<?x?xi8, #hal.descriptor_type<storage_buffer>>
  to memref<?x?xi8, strided<[?, 1]>, #hal.descriptor_type<storage_buffer>>   // OUTPUT, strides [1,1] = contiguous
%26 = iree_codegen.load_from_buffer %subview : ... -> tensor<?x?xi8>
%27 = scf.forall (%arg0, %arg1) = (0,0) to (%22, %25) step (64, 64)
        shared_outs(%arg2 = %26) -> (tensor<?x?xi8>) {
    ...
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %30 into %arg2[%arg0, %arg1] [%28, %29] [1, 1]   // writeback strides [1,1]
    }
} {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}
iree_codegen.store_to_buffer %27, %subview : tensor<?x?xi8>
        into memref<?x?xi8, strided<[?, 1]>, #hal.descriptor_type<storage_buffer>>   // store strides [1,1]
return
```

Direct observations:

- The store op kind is **`iree_codegen.store_to_buffer`** (control_dump:25809).
- The writeback inside the forall is a **`tensor.parallel_insert_slice … [1, 1]`**
  (control_dump:25806) — unit strides.
- The output store `%subview` has **strides `[1,1]`** (control_dump:25781 and
  :25809 — a contiguous subview, type `strided<[?, 1]>`).
- **Region strides `[1,1]` and output store strides `[1,1]` are identical** —
  they agree on *every* axis. There is no transposition.

### 3.2 Why it compiles — in-place aliasing, *not* tensor fusion

At the verifier input (`control_dump.mlilr:27459-27479`), the control IR is:

```mlir
%subview = memref.subview %assume_align[0, 0] [%22, %25] [1, 1] : ...   // contiguous output binding
scf.forall (%arg0, %arg1) = (0,0) to (%22, %25) step (64, 64) {
    %subview_0 = memref.subview %subview[%arg0, %arg1] [%26, %27] [1, 1] : ...
    scf.for ... {
        vector.store %cst, %subview_0[%arg2, %arg3] ...        // fill writes DIRECTLY to the output binding
        linalg.generic outs(%subview_1 ...) { yield %c1_i8 }   // also direct to the output binding
    }
} {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]}
return
```

Two things happened during bufferization:

1. The forall's `shared_outs` (`%26 = load_from_buffer %subview`) was
   **aliased in-place to the output binding `%subview`**, so the tiled fill
   writes straight into the global buffer inside the forall — no separate
   result tensor.
2. The `store_to_buffer %27, %subview` therefore degenerated to a copy of
   `%subview` onto itself, which copy-elimination removed. By the verifier
   input there is **no write to a global outside the forall** (the function
   body ends at `return` immediately after the forall, control_dump:27479).

This is why `VerifyWorkgroupDistributionPass` passes: the verifier's PreOrder
walk (`VerifyWorkgroupDistribution.cpp:48-56`) skips the contents of the
workgroup-mapped forall, and finds no remaining write to a global
`storage_buffer` at dispatch scope (`:57-76`).

**The control does not exercise tensor-level fusion of a strided store at
all.** It compiles purely because one-shot bufferization could alias the
matching unit-stride layouts in-place. Fusion was a no-op here (the only
outside-forall consumer was the non-`TilingInterface` `store_to_buffer`).

### 3.3 Contrast — the two-axis (transposed) failing case

`dump.mlir:31648-31679` (same pass boundary):

```mlir
%subview = memref.subview %33[0, 0] [%29, %30] [2, 1]                // OUTPUT store: strides [2,1]  (dim-0 strided)
    : ... to memref<?x?xi8, strided<[?, 1]>, #hal.descriptor_type<storage_buffer>>
%35 = iree_codegen.load_from_buffer %32 -> tensor<?x?xi8>            // load WHOLE source (binding 0, ReadOnly)
%extracted_slice = tensor.extract_slice %35[0, 0] [%29, %34] [1, 2]  // fill REGION: strides [1,2]  (dim-1 strided)
%36 = scf.forall (...) shared_outs(%arg2 = %extracted_slice) -> ... { ..fill.. }
%inserted_slice = tensor.insert_slice %36 into %35[0, 0] [%29, %34] [1, 2]   // OUTSIDE forall, dim-1 strided
iree_codegen.store_to_buffer %inserted_slice, %subview : ... into strided<[?, 1]>  // OUTSIDE forall, dim-0 strided
```

The mismatch is **transposed**: the fill region / writeback is strided on
**dim-1** (`[1,2]`), while the output store is strided on **dim-0** (`[2,1]`).
They agree on *no* strided axis. One-shot bufferization cannot alias a
dim-1-strided source tensor into a dim-0-strided destination memref, so it
materializes the full result and emits a bare
`linalg.generic { yield %in }` copy of `#hal.descriptor_type<storage_buffer>`
at dispatch scope — exactly the op class `VerifyWorkgroupDistributionPass:57-76`
rejects.

### 3.4 Precisely what must change to make the two-axis case land in-place like the control

For the failing writeback to be representable *inside* the workgroup forall
the way the control's is, the tiled body would have to emit, per workgroup,
**a strided `parallel_insert_slice` / store whose per-dimension strides are
not all 1** — e.g. a `[2,1]` write into the output binding. Concretely, the
desired post-fix shape (sketched, mirroring the control's in-place pattern)
would be:

```mlir
%subview = memref.subview %33[0,0][%29,%30][2,1] : ... to strided<[?,1]>   // dim-0-strided output
scf.forall (%arg0, %arg1) = (0,0) to (%29div2, %30) step (64,64) {
    %tile = memref.subview %subview[%arg0*2, %arg1] [64, 64] [2,1] : ...   // dim-0-strided tile of the OUTPUT
    // fill %tile in place (read-modify-write the existing rows)
    linalg.generic outs(%tile ...) { yield %c1_i8 }
} {mapping = [workgroup y, workgroup x]}
return
```

The single new capability this requires — which **does not exist today** — is
the ability for the workgroup tiler to (a) compute a workgroup tile whose
memory footprint is a *strided subview* of the output binding, and (b) emit a
writeback whose `static_strides` are the destination's actual per-dimension
strides rather than the literal `1` that `TileUsingInterface.cpp:1006-1010`
hardcodes. The control proves the *plumbing* (in-place aliasing → verifier
passes) works; the transposed case proves the *stride axis* is the only thing
the plumbing cannot currently express.

---

## 4. What must change (concrete code-level change list)

Framed as a generalization of §3, in dependency order:

### 4.1 IREE-side (necessary, trivial, *insufficient on its own*)

1. **Widen the candidate seed in `fuseConsumersIntoForall`.** Replace/augment
   the `dyn_cast<tensor::ParallelInsertSliceOp>` at
   `TileAndFuseUtils.cpp:141` to also accept
   `tensor::InsertSliceOp` (so the outside-forall
   `tensor.insert_slice %36 into %35[…][1,2]` becomes a seed), and relax the
   `filterFn` contract so a genuine consumer (not already in
   `tiledAndFusedOps`) can be fused. This alone changes ~10 lines.

2. **But `:215-216` will then call `tileAndFuseConsumer` on a strided
   consumer, which fails at `TileUsingInterface.cpp:2313-2317`.** So 4.1 is a
   no-op without 4.2.

### 4.2 Upstream `TilingInterface` + SCF (the load-bearing, high-blast-radius change)

The first-pass framing — "thread strides through `YieldTiledValuesFn` and stop
hardcoding `resultStride`" — is necessary but **insufficient**, because it only
gives SCF a place to *emit* a stride, not a *source* for one (§2.4.1). The true
load-bearing prerequisite is extending the `TilingInterface` contract, then SCF,
then every implementor:

3. **Add strides to the `TilingInterface` contract.** Give `getResultTilePosition`
   a `resultStrides` out-parameter, and pass/return strides through
   `getTiledImplementationFromOperandTiles` / `getIterationDomainTileFromOperandTiles`
   / `generateResultTileValue`, so an op can report that its result tile occupies
   a *strided* region of the destination. This edits
   `mlir/include/mlir/Interfaces/TilingInterface.td` and is consumed by **every**
   `TilingInterface` implementor (linalg, tensor slice ops, pack/unpack, …).
4. **Update the implementors to populate strides.** Linalg
   (`TilingInterfaceImpl.cpp:236-258`) computes none today; pack/unpack (`:1029`,
   `:1499`) and tensor slice ops likewise. Without this, SCF still has nothing
   to emit — 5-6 would be dead code.
5. **Thread strides through `YieldTiledValuesFn`** (its `:336-340` typedef returns
   only `tiledOffset`/`tiledSizes`) and **stop hardcoding `resultStride` at all
   six sites** — `:447-448`, `:616-617`, `:951-955`, `:1006-1007`, `:1565-1566`,
   `:2366-2367` (the first-pass list named only three). Use the interface-provided
   strides instead of `rewriter.getIndexAttr(1)`.
6. **Replace the two correctness rejections with stride propagation.**
   `:2313-2317` (consumer) and `:1502-1504` (producer) must *carry* the
   candidate's strides (`candidateSliceOp.getMixedStrides()`, already available
   at `:2311`) into the tiled body. They are correctness guards that can only be
   removed once 3-5 land.

This touches `TilingInterface.td` (interface contract), `TileUsingInterface.cpp`
(foundation file used by every SCF-tiling dialect — linalg, tensor, vector,
affine, all of IREE), and every `TilingInterface` implementor. It is not
IREE-local, and even merged, the lowered strided code then hits the
*vectorization*-layer sibling of the same wall — LLVM issue #51660 (§8.1).

### 4.3 IREE-side (consumes 4.2)

7. **Anchor selection (`TileDispatchUsingForall.cpp:67-76`).** Optionally
   allow the strided `tensor.insert_slice` to be (co-)anchored so the
   tiled forall body directly contains the strided write, rather than relying
   on consumer fusion to pull it in. A `tensor.insert_slice` *is*
   `TilingInterface`, but has no lowering config today, so it would need a
   synthesized distribution config, or the existing fill-anchor + 4.1 fusion.
8. **Verify `verifyComputeOpsAfterDistribution` still holds**
   (`TileDispatchUsingForall.cpp:196-209`): once the strided insert lives
   inside the forall, no `computeOp` remains outside, so this passes.

### 4.4 Sketched *after* IR (the goal state for the failing dispatch)

After 4.1+4.2, the failing dispatch should lower to the in-place shape of
§3.4 — a single workgroup forall writing the dim-0-strided output directly,
with the source load co-distributed (see §5.2), and no dispatch-scope copy.
The verifier then passes for exactly the reason the control's does (§3.2).

---

## 5. The four NOTE §4 open questions, each answered

### Q1 — Dilation expressiveness: does a "write every k-th element" primitive exist in CPU codegen?

**No. There is no such primitive; it is the core capability to add.** Every
stride-handling site in IREE CPU codegen **assumes unit strides and bails
otherwise**:

- `CombineLayoutTransformation.cpp:220-223`: `if (!areAllConstantIntValue(extractSliceOp.getMixedStrides(), 1)) { return failure("non-unit strides are not supported"); }`
- `ReshapePatterns.cpp:415-417` and `:622-624`: bail with "found a non-1
  stride" / "expected unit strides" on `store_to_buffer`.
- `IREECodegenCanonicalizer.cpp:31-32`: only folds subviews with unit strides.
- `TensorDynamicDimAnalysis.cpp:110-113`: aborts unless flow-load strides are 1.

The `dilation`/`strides` hits that *do* exist are all **linalg convolution
window strides** (e.g. `linalg.conv_2d_nhwc_hwcf {strides = dense<2>}`,
`gpu_create_fast_slow_path.mlir:38`) — a semantic property of the conv op
handled *inside* the structured op, unrelated to workgroup-distributed memory
writes. `iree_linalg_ext.im2col strides=` (`gpu_apply_derived_thread_config.mlir:175`)
is likewise a window stride. None of these provide an affine "workgroup i maps
to memory offset `i*k`" dilation for a distributed store.

The capability to add is an **affine per-dimension dilation factor** on the
workgroup-tile → memory-offset mapping, plumbed through the upstream tiler
(§4.2 items 3-5). `[INFERENCE]` Uniform doubling `[2,…]` is the realistic
first target; arbitrary per-dim strides `[a,b]` need full affine stride
composition and are a strict superset.

### Q2 — Read-modify-write / zero preservation: must the input load be co-distributed?

**Yes, the load must be co-distributed; the output cannot be assumed
pre-zeroed.** The failing dispatch is a read-preserve-write:
`dump.mlir:31650` does `%35 = iree_codegen.load_from_buffer %32` where `%32`
is binding 0 (`ReadOnly`, `dump.mlir:31646`) — i.e. it materializes the
**entire source** to build the result, then extracts the strided region
(`:31651`) and writes it back strided into the output (`:31679`). For
`m[mask]=True`, the non-masked cells must retain their source values, so a
strided write that touches only `[even,even]` cells is only correct if the
remainder is preserved — which is exactly why the dispatch loads the whole
source rather than overwriting.

If the store is distributed per-workgroup (the §3.4 goal), each workgroup must
read its own tile of the source (read-modify-write its strided subregion),
mirroring the control's in-place pattern where each workgroup subviews the
shared output `%subview` and reads/writes its tile
(`control_dump.mlir:27463-27475`). The load and store must therefore be
**co-distributed**; an "output is pre-zeroed" assumption would be a
correctness regression.

### Q3 — Generality: arbitrary per-dim strides, or uniform doubling first?

The realistic input variety is encoded by `flow.dispatch.tensor.store`'s
strides attribute. `[INFERENCE]` The common case in real models is small
**uniform** factors (doubling/halving from strided slice/scatter, dilated
writes), not adversarial coprime `[a,b]`. The principled end state supports
arbitrary per-dim strides, but the tractable first milestone is uniform
doubling `[2,2]` (and `[2,…]` generally) — enough to unblock the motivating
`m[0::2,0::2]=True` class. Arbitrary strides additionally require the affine
stride composition of Q1 and more careful tile-size divisibility checks
(`tileDividesIterationDomain`, `TileUsingInterface.cpp:221-233`, currently
only used for the *loop* stride, not a memory-write dilation).

### Q4 — Verifier interaction: does the fix actually satisfy `VerifyWorkgroupDistributionPass`?

**Yes, and no other invariant is violated, provided the store lands lexically
inside the workgroup forall.** Re-reading `VerifyWorkgroupDistribution.cpp`:

- The early-exit at `:41-43` requires *some* workgroup-mapped forall to exist
  (the failing dispatch has one — `dump.mlir:31677` — so verification is
  active).
- The PreOrder walk at `:48-56` **skips** the contents of any
  workgroup-mapped forall.
- The write check at `:57-76` flags only a `MemoryEffects::Write` on a
  global-address-space (`#hal.descriptor_type<storage_buffer>`) operand
  *outside* such a forall.

So once the strided store is inside the forall, the walk skips it → passes
— exactly the control's situation (§3.2). Two related invariants to confirm:

- **`shared_outs` aliasing:** the control shows one-shot bufferization
  aliasing `shared_outs` to the output binding in-place is *legal and
  expected* (`control_dump.mlir:27460-27478` — forall writes directly to the
  global). No invariant forbids it.
- **`verifyComputeOpsAfterDistribution` (`TileDispatchUsingForall.cpp:196-209`):**
  walks for any `computeOp` (`isComputeOp`, i.e. `TilingInterface`) left
  outside a forall, except `linalg::PackOp` (`:192-194`). Once the strided
  `tensor.insert_slice` (a `computeOp`) is fused inside, nothing violates it.
  Note `store_to_buffer` is **not** a `computeOp`, so even today a stranded
  `store_to_buffer` does *not* trip this check — the failure is purely the
  bufferization-emitted bare copy, not `store_to_buffer` itself.

---

## 6. Risks, blast radius, and prerequisites

**Blast radius of widening `fuseConsumersIntoForall` (IREE-side, §4.1):**
small in lines, but the pass is `TileAndDistributeToWorkgroupsUsingForallOp`,
the **sole** CPU workgroup-distribution pass — every llvm-cpu dispatch flows
through it. Downstream consumers of the "only `ParallelInsertSliceOp` is
fused" contract:

- `verifyComputeOpsAfterDistribution` (`TileDispatchUsingForall.cpp:196-209`)
  and the runOnOperation failure path at `:376-382` assume fusion either
  succeeds fully or is an allowed-to-fail op (`linalg::PackOp`, `:192-194`).
  Admitting strided consumers that then fail upstream
  (`TileUsingInterface.cpp:2313-2317`) would turn currently-passing dispatches
  into hard failures unless guarded.
- `fuseProducersOfSlices` (`TileAndFuseUtils.cpp:31-66`, called at
  `TileDispatchUsingForall.cpp:388`) consumes the new-fusion-opportunities
  queue; its assumptions about slice shape are also unit-stride-oriented.

**Blast radius of the upstream change (§4.2):** very high.
`TileUsingInterface.cpp` is foundational — linalg, tensor, vector, affine and
all of IREE tile through `tileUsingSCF`/`tileConsumerAndFuseProducersUsingSCF`/
`tileAndFuseConsumer`. Changing writeback/dest-slice stride semantics risks
silently miscompiling any tiling that today relies on the unit-stride
invariant (the rejections at `:2313-2317`/`:1502-1504` exist as correctness
guards). This is also an **upstream-MLIR acceptance battle**: IREE vendors a
checkout at `third_party/llvm-project/` but the change must land upstream to
avoid a fork. `[INFERENCE]` That negotiation is likely the schedule-driving
item, more than the code itself.

**Prerequisites:**
- Either an upstreamable stride-aware SCF tiling patch (§4.2), **or** an
  explicit decision to maintain an IREE-local tiling fork for strided
  writebacks (strictly worse maintenance story).
- Co-distribution of the source load (Q2) so the distributed store is a
  correct read-modify-write, not an overwrite.
- A divisibility/peeling story for tile sizes vs. the dilation factor (Q3).

**Independent effort rating:** HARD+, **>3 days**. The IREE filter widening
is hours; the upstream SCF stride-threading (4 sites + 2 guard removals +
`getIterationDomainTileFromOperandTiles`/`getResultTilePosition` interaction)
plus upstream review plus co-distribution plus testing is the multi-week
component. The expert review's "HARD (Large, 3d+), high risk — touches the
core workgroup-distribution strategy used by EVERY dispatch" is *correct in
spirit*; this investigation adds that the load-bearing difficulty is upstream
(`TileUsingInterface.cpp`), not the IREE pass, and is therefore somewhat
larger and more political than a pure IREE change.

---

## 7. Execution order — what to fix, and in what sequence

This section consolidates the dependency ordering. Two tracks are cleanly
separated, because (per §8.1) **compilation and performance have different
blockers**: the compile path is the hard gate; #51660 is a performance finisher,
*not* a prerequisite.

### 7.1 Why #51660 is NOT a prerequisite

The two layers fail in *different ways*:

- **This approach's issue (tiling/fusion, §2.4) is a hard compile error.**
  `VerifyWorkgroupDistributionPass` rejects the dispatch-scope `storage_buffer`
  write; compilation aborts. This gates everything below it in the pipeline
  (tiling → bufferization → vectorization → codegen).
- **#51660 (vectorization, §8.1) is a graceful degradation, not an error.** Its
  own wording — vectorization is *"disabled"* for non-contiguous dense access —
  means the failure mode is **scalar fallback**: correct code, just slow. It is a
  *performance* problem, not a correctness or compilation problem.

Consequence:
- **Fixing #51660 without fixing the tiler changes nothing** — the code still
  dies at the tiler; the vectorizer never sees the strided store.
- **Fixing the tiler without #51660 makes it compile** — via scalar strided
  stores (correct, unvectorized).

So #51660 is a *follow-on performance* dependency, not on the compile critical
path. The `TilingInterface`+SCF change is the hard gate that must go first.

### 7.2 The compile critical path (dependency order, innermost first)

| # | What | Where | Why this order |
|---|---|---|---|
| **1** | Add strides to the `TilingInterface` **contract** — `resultStrides` out of `getResultTilePosition`; strides through the operand-tile methods | `mlir/include/mlir/Interfaces/TilingInterface.td` | Root (§2.4.1). Nothing below has a stride source without it. |
| **2** | Make implementors **populate** those strides | linalg `TilingInterfaceImpl.cpp:236-258`, tensor slice ops, pack/unpack | SCF can't emit what no op reports (§4.2 item 4). |
| **3** | SCF: thread strides through `YieldTiledValuesFn` (`:336-340`), stop hardcoding at all **six** sites, flip the 2 rejections reject→propagate | `TileUsingInterface.cpp` | Consumes 1+2; high-blast-radius file (§4.2 items 5-6). |
| **4** | IREE: widen the fusion filter to seed the strided `tensor.insert_slice` | `TileAndFuseUtils.cpp:141` + `filterFn :154-155` | No-op until 1-3 land; ~10 lines (§4.1). |
| **5** | **Co-distribute the source load** (read-modify-write) so non-masked cells are preserved | IREE workgroup distribution | Correctness, not just compilation (§5.2/Q2). Parallel to 3-4. |

Steps 1-3 are upstream MLIR (IREE vendors llvm-project at
`third_party/llvm-project/`); 4-5 are IREE-local. The decision point at 1-3 is
the **upstream-vs-local fork**: land upstream (principled, but an
upstream-acceptance battle) vs. an IREE-local tiling fork (faster to ship, worse
maintenance, diverges from upstream).

### 7.3 Then — performance (#51660), strictly after the above compiles

| # | What | Where |
|---|---|---|
| **6** | Vector-dialect strided dense load/store so the strided store vectorizes instead of scalar-falling-back | LLVM #51660 (§8.1) |

### 7.4 What to actually do first

**Phase 0 — de-risk (~1 day, before committing to weeks):** prototype steps 1+3
*in isolation* on the vendored llvm-project with an MLIR lit test — fuse a
`[1,2]`-strided `tensor.insert_slice` into an `scf.forall` and assert the
writeback preserves `[1,2]`. Needs none of #51660 and none of the IREE plumbing.
Converts "can upstream represent a strided writeback at all?" from `[INFERENCE]`
to a binary yes/no. If it can't, Approach 1 is dead — pivot to Approach 2/3,
having spent a day, not weeks.

**If Phase 0 passes → Phase 1** (1-3, the upstream-or-local fork) **→ Phase 2**
(4+5; now it compiles) **→ Phase 3** (#51660; now it's fast).

**Recommendation:** treat Approach 1 as the *target* architecture, but Phase 0
(not the full effort) is the first move.

> *Provenance note:* the prior version of this section sketched the prototype as
> "thread strides through `YieldTiledValuesFn` and replace the **four**
> unit-stride hardcodings." That predates the §2.4.1 root-cause finding: the
> prototype must start at the `TilingInterface` contract (step 1), and there are
> **six** hardcode sites, not four.

---

## 8. Upstream corroboration — LLVM issue #51660 and the stride-assumption survey

*(Added 2026-06-29 from a second pass over the checked-out
`third_party/llvm-project/` MLIR tree.)*

### 8.1 LLVM issue #51660 — the *vectorization*-layer sibling of this wall

[LLVM #51660](https://github.com/llvm/llvm-project/issues/51660) — *"Implement
non-unit stride dense vectorization"* — is **not** a sparse-data issue (despite
its `sparsetensor` label): it is explicitly about **dense arrays** — *"Implement
strided load/stores on dense arrays, so that vectorization does not need to be
disabled when the access pattern is non-contiguous for the dense data
structures."* It is filed under `sparsetensor` only because its entry point,
`denseUnitStrides()`, lived in `Sparsification.cpp` as the gate that checked
whether a dense sub-region had unit strides before vectorizing it.

This is **the same class of limitation as this approach's tiling-layer wall, one
layer down.** Where §2.4 shows the *tiling/fusion* layer cannot emit a strided
dense writeback, #51660 shows the *vectorization* layer (the vector dialect's
dense load/store) cannot represent a strided dense access and so disables
vectorization. The issue author (aartbik) names the prerequisite explicitly:
*"requires adding vector dialect support first (for the non-unit stride load and
stores)."* **Implication for Approach 1:** even if the `TilingInterface`+SCF
change (§4.2) succeeds in *emitting* a strided writeback, the lowered code then
reaches #51660's wall at vectorization time unless the vector dialect also gains
strided dense load/store. **Crucially this is a *performance* wall, not a
compile wall** — #51660's failure mode is graceful (vectorization disabled →
scalar strided stores → correct but slow), so it is *not* on the compile critical
path; it is a follow-on performance dependency (see §7.1 for why it is not a
prerequisite). The two issues are nonetheless complementary evidence that
non-unit-stride dense memory access is a broad, cross-layer MLIR gap.

*Provenance:* `denseUnitStrides()` was added 2021-05-03 (`a2c9d4bb04a9`,
"Introduce proper sparsification passes"), the issue was filed 2021-10-26, the
function was removed 2022-10-18 (`26eb2c6b42f7`, "remove vector support in
sparsification" — vector codegen relocated to separate passes), and the issue is
**still open** as of 2026-03-17 with an active GSoC proposal. ~4.5 years open —
i.e. genuinely unsolved, not neglected.

### 8.2 The unit-stride assumption is systemic, not SCF-local

A survey of the checked-out tree shows the same `isOneInteger`/unit-stride bail
recurs across every tiling/fusion/bufferization layer, all via the one shared
helper (`StaticValueUtils.h:38`):

- `Linalg/Transforms/SwapExtractSliceWithProducerPatterns.cpp:31` and `:99` —
  two bails carrying the explicit comment *"`TilingInterface` currently only
  supports strides being 1"* (the smoking gun for §2.4.1).
- `Linalg/Transforms/DataLayoutPropagation.cpp:1480-1483` — *"propagation of
  strided extract slice is unsupported."*
- `Tensor/Transforms/ReshapePatterns.cpp:577` — *"Only unit stride is supported."*
- `Tensor/Transforms/BufferizableOpInterfaceImpl.cpp:672-674` — `allStridesOne`
  gate on `insert_slice` in-place aliasing (directly relevant: this is *why* the
  transposed case cannot be aliased in place, §3.3).
- Plus the IREE-side bails already catalogued in §5.1
  (`CombineLayoutTransformation.cpp:220-223`, `ReshapePatterns.cpp:415-417,622-624`,
  `IREECodegenCanonicalizer.cpp:31-32`, `TensorDynamicDimAnalysis.cpp:110-113`).

**Read together with #51660:** non-unit-stride dense access is rejected at
*tile-and-fuse* (SCF), at *producer-swap* (linalg), at *layout propagation*
(linalg), at *reshape* (tensor), at *bufferization in-place aliasing* (tensor),
at *IREE layout transforms*, and at *vectorization* (vector dialect). It is a
single conceptual assumption, copy-pasted across the whole structured-ops stack.
This materially raises Approach 1's blast radius beyond "patch SCF": a clean fix
touches the `TilingInterface` contract (§2.4.1) and rippled implementors across
at least three dialects, with the vectorization layer still open upstream.

---

## Appendix A — citation index (all personally opened)

**IREE compiler:**
- `Codegen/Common/TileAndFuseUtils.cpp:112-247` — `fuseConsumersIntoForall`;
  `:141` dyn_cast gate; `:154-155` filterFn gate; `:215-216` `tileAndFuseConsumer` call.
- `Codegen/Common/TileDispatchUsingForall.cpp:61-142` — `getTiledAndDistributionInfo`
  (anchor = last compute op with workgroup config, `:67-76`);
  `:196-209` `verifyComputeOpsAfterDistribution`; `:192-194` allowed-to-fail;
  `:228-434` `runOnOperation`; `:353-355` producer tile+fuse;
  `:370-375` consumer-fusion call + filter.
- `Codegen/Common/VerifyWorkgroupDistribution.cpp:29-84` — full verifier
  (`:41-43` early-exit, `:48-56` forall skip, `:57-76` global-write check).
- `Codegen/Dialect/Codegen/IR/IREECodegenOps.td:206-228` (`LoadFromBufferOp`),
  `:230-250` (`StoreToBufferOp` — `MemoryEffectsOpInterface` only, no `TilingInterface`),
  `:256-269` (`InnerTiledOp` is the `TilingInterface` one).
- `Codegen/Utils/Utils.cpp:980-982` — `isComputeOp`.
- `Codegen/Common/BufferizeDispatchTensorLoadStore.cpp:58, :70-88` —
  `store_to_buffer`/`load_from_buffer` created from flow load/store.
- `Codegen/Common/CombineLayoutTransformation.cpp:220-223`,
  `Codegen/Common/ReshapePatterns.cpp:415-417,622-624`,
  `Codegen/Common/IREECodegenCanonicalizer.cpp:31-32`,
  `Codegen/Common/TensorDynamicDimAnalysis.cpp:110-113` — non-unit-stride bails.

**Upstream MLIR — `TilingInterface` contract (the root cause, §2.4.1):**
- `include/mlir/Interfaces/TilingInterface.td:64-301` — five core methods, none
  takes/returns strides (`getResultTilePosition` at `:118-162`; the two
  operand-tile methods at `:202-301`).
- `lib/Dialect/Linalg/Transforms/TilingInterfaceImpl.cpp:236-258` — linalg
  `getResultTilePosition` builds offsets/sizes only; strides dropped.
- `lib/Dialect/Tensor/Transforms/SwapExtractSliceWithProducerPatterns.cpp:31,99`
  — *"TilingInterface currently only supports strides being 1"* (smoking gun).

**Upstream MLIR (`lib/Dialect/SCF/Transforms/TileUsingInterface.cpp`):**
- `:2521-2570` `tileAndFuseConsumer` (`:2524-2527` TilingInterface requirement;
  `:2554-2567` candidate = producing insert-slice-like op).
- `:2437-2443` candidates must be all `InsertSlice` or all `ParallelInsertSlice`.
- `:2205-2416` `tileAndFuseConsumerOfSlicesImpl`;
  `:2224-2234` DPS requirement; `:2313-2317` consumer stride rejection;
  `:2366-2367` dest-extract unit-stride hardcode.
- Six unit-stride hardcode sites (verified by grep): forall writeback
  `:616-617` (initial) + `:1006-1007` (fusion add-init); `scf.for` writeback
  `:447-448` (initial) + `:951-955` (fusion add-init); DPS dest extract
  `:2366-2367` (consumer) + `:1565-1566` (producer).
- `:1502-1504` producer-fusion stride rejection; `:221-233` `tileDividesIterationDomain`.
- Shared helper: `include/mlir/Dialect/Utils/StaticValueUtils.h:38` `isOneInteger`.

**Upstream MLIR — sibling-dialect unit-stride bails (§8.2):**
- `lib/Dialect/Linalg/Transforms/DataLayoutPropagation.cpp:1480-1483`.
- `lib/Dialect/Tensor/Transforms/ReshapePatterns.cpp:577`;
  `lib/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.cpp:672-674`.

**Upstream issue:**
- [LLVM #51660](https://github.com/llvm/llvm-project/issues/51660) — non-unit
  stride *dense* vectorization (vector-dialect layer); open since 2021-10-26;
  `denseUnitStrides()` added `a2c9d4bb04a9` (2021-05-03), removed
  `26eb2c6b42f7` (2022-10-18); GSoC proposal active 2026-03-17.

**Repro IR:**
- Failing (two-axis): `editor/rcd_lowpass_llvm_cpu_repro/dump.mlir:31646-31679`
  (pre-bufferize), `:31683+` (post-bufferize bare copy).
- Control (single-axis): `editor/rcd_lowpass_llvm_cpu_repro/control_dump.mlir:25781-25809`
  (pre-bufferize), `:25814-25889` (post-bufferize), `:27459-27479` (verifier input).
- Repros: `reduced_reproducer.mlir` (`m[0::2,0::2]`), `control_single_stride.mlir` (`m[:,0::2]`).
