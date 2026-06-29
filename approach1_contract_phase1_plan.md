# Approach 1 — Phase 1 Implementation Plan (CONTRACT-BASED, v3 — A3 polish)

> **Status:** DRAFT v3 (A3 polish), execution-ready (R2 verdict: CONVERGED;
> m1–m4 polish applied, no remaining must-fixes). This plan
> **replans Approach 1 around the `TilingInterface.td` contract change.** It
> supersedes `approach1_phase1_impl_plan.md` (SCF-only consumer fusion —
> REFUTED; see `expert_review_phase1_plan.md`).
>
> **v2 (A2) changes — see `approach1_contract_rework_a2_changes.md` for the
> per-finding changelog.** This pass folds in R1's review
> (`approach1_contract_plan_review_r1.md`) and the orchestrator's re-directives.
> Load-bearing corrections: **(F1)** the per-tile writeback uses
> `resultSizes = iterSizes` (the *source* tile shape), NOT `iterSizes*stride` —
> `verifyInsertSliceOp` (`TensorOps.cpp:2885-2896`) checks `sizes` against the
> source via `inferResultType(dstType, sizes)`, so the dest span
> `offset+(size-1)*stride` is implicit; **(F2)** the Case-1 anchor IR is
> `insert_slice … [0,0][2,4][2,1]` (sizes = source shape); **(A/B reframe)** the
> PRIMARY G3/G4 proof is the full-source 1-tile *genuine scatter*
> (`%tile<2x4>` → dest rows {0,2}); a size-1 strided dim is a *vestigial*
> stride (element j=0 lands at `offset`, identical to unit stride) and is
> demoted to the offset-placement sub-case. Added: blast-radius containment
> (§4a), the real EXEC lowering trace (§8), rank-reduction rule (§11 #3),
> numThreads promotion to core G4 (§7/§10), and a Reasoning section (§12).
>
> **v3 (A3) changes — see `approach1_contract_rework_a3_changes.md` for the
> per-finding changelog.** R2's verdict was CONVERGED; this pass applies the four
> remaining execution-risk fixes (m1–m4) without re-opening the settled mechanism:
> - **m1 (§4a):** the marker-gate point is corrected — `getIterationDomain`
>   returns `SmallVector<Range>` (`TilingInterface.td:80-85`) and *cannot* return
>   failure, so the gate lives on `getTiledImplementation` (`:107`) /
>   `getResultTilePosition` (`:149`); the full bail path is re-traced and verified.
> - **m2 (§8):** the EXEC pipeline adds `-expand-strided-metadata
>   -finalize-memref-to-llvm` (the pass owning `MemRefCopyOpLowering`,
>   `MemRefToLLVM.cpp:1140`); the stale `-convert-memref-to-llvm` name (no such
>   pass exists — `Passes.td:994-995`) is corrected; R2 §4's resolved-positive
>   finding (strided copy is NOT rejected, `:1263-1277`) is folded in.
> - **m3 (§3):** the G3 FileCheck is made fold-robust (match `[0,0][2,4][2,1]`
>   geometry under EITHER `insert_slice`/`parallel_insert_slice`), with
>   no-canonicalize / 2-tile fallbacks.
> - **m4 (§4a):** the marker-gate + allow-list containment is stated honestly as
>   best-effort (partial-forall-leak checkpoint, bypass-via-ungated-methods,
>   best-effort `tileToThreads` bail), with the allow-list as the hard firewall.
>
> **Grounding method:** every `file:line` below was personally opened in the
> `~/Developer/iree` / `third_party/llvm-project` tree during planning. Reasoned
> (not directly observed) claims are marked `[INFERENCE]`. No code changed, no
> builds run.
>
> **One-line thesis:** strides can only be *introduced* at **initial tiling**
> (`generateLoopNestUsingForallOp:616-617`), and the only op whose *result* is a
> strided destination is `tensor.insert_slice`. Therefore the stride source is
> the insert_slice's own geometry, made reachable at initial-tiling time by
> giving `tensor.insert_slice` a `TilingInterface` impl whose
> `getResultTilePosition` returns a **strided result position** — flowing through
> a new `resultStrides` out-param on `getResultTilePosition`, the
> `GenerateTiledBodyFn` channel, and into the `:616-617` writeback. This is
> **initial tiling of the slice op**, never consumer fusion of it.

---

## 0. What survives from the refuted work (reuse, don't repeat)

From `expert_review_phase1_plan.md` (§4, §5) and the superseded plan:

- **R2 — GREEN asserts the COMPLETE per-tile writeback IR** (offsets AND sizes
  AND strides), hand-derived for a small static case. A `[2,1]`-only FileCheck
  false-passes while the offset is wrong (silent-miscompile zone). **Reused
  verbatim** as the acceptance contract for every GREEN.
- **Offset math** (expert-confirmed, §4 of the review): for an identity-map
  writeback the per-tile strided-dim offset is `iv * tileSize * stride`
  (`= iv*2` for tile-size-1, stride-2). Generalized here to
  `offset[d] = insert_base[d] + iter_offset[d] * stride[d]` because the stride
  source is now the slice op (not a post-multiply hack on a unit-stride
  candidate). `iter_offset[d]` already carries the effective tile size
  (`iv * ceilDiv(range, numThreads)` under `numThreads`), so the
  `iv*T*S` composition is correct *by construction* (§7).
- **Static stride 2 first** (2× write); arbitrary/coprime strides deferred.

**What is NOT reused (the 3 refuted flaws — do not repeat):**
1. `tensor.insert_slice` is NOT `TilingInterface` today
   (`TensorTilingInterfaceImpl.cpp:311-316` registers only `PadOp`). The
   refuted plan assumed it could be a *fused consumer*. This plan instead
   **makes it the tiled anchor** (initial tiling), which is a different and
   valid topology.
2. Consumer fusion reads the loop-*internal* unit-stride candidate
   (`getProducingParallelInsertSlice`, `TileUsingInterface.cpp:2487`), never the
   external strided store — it preserves but cannot create strides
   (`expert_review_phase1_plan.md` §3 FLAW 3). **No stage here relies on
   consumer fusion of `insert_slice`.**
3. `transform.structured.fuse_into_containing_op` does *producer* fusion
   (`LinalgTransformOps.cpp:1335-1336`). **Not used as the driver.**

---

## 1. The crux, solved — the STRIDE SOURCE at initial tiling

### 1.1 Why the fill can never be the stride source

The IREE anchor today is `linalg.fill` (`TileDispatchUsingForall.cpp:67-76`;
`isComputeOp` = `TilingInterface | UKernelOpInterface`, `Utils.cpp:980-982`).
Tiling the fill drives `generateLoopNestUsingForallOp:616-617` via the call
chain in §1.3. But the fill's **result is contiguous** in its own tensor space:
`linalg::getResultTilePosition` (`TilingInterfaceImpl.cpp:235-259`) uses
`computeSliceParameters` with the indexing map; for `fill` (identity map) it is a
pure passthrough of `offsets`/`sizes` — there is no stride to return, and the
interface has no stride out-param anyway (`TilingInterface.td:118-162`).
Confirmed: linalg indexing maps are projected permutations
(`:274`, the `isProjectedPermutation()` check in
`getIterationDomainTileFromResultTile:261-288`; `getMappedOffsetAndSize:156-208`
enforces `AffineDimExpr` at `:169-171`). **A projected permutation cannot
express stride-2.** So no amount of contract editing makes the *fill* return a
strided result position.

The `[2,1]` lives on the **external `tensor.insert_slice` destination**, whose
result *is* the strided-write tensor. The stride is a property of the slice op,
not the fill.

### 1.2 The chosen mechanism — (c) make `tensor.insert_slice` a `TilingInterface` anchor

Evaluated against the actual code:

| Candidate | Verdict | Evidence |
|---|---|---|
| **(a)** thread destination stride as INPUT to `getResultTilePosition` | **REJECT.** `getResultTilePosition` (`TilingInterface.td:118-162`) has no destination operand; SCF's `tileUsingSCF` builds the body lambda (`TileUsingInterface.cpp:1158-1226`) from the *anchor op alone* + `initTensors` (`createInitialTensorsForTiling:726`, which for a parallel op calls `tensor::getOrCreateDestinations:734`). There is no "external destination stride" in the tiling input to thread. Adding one means SCF invents a new caller-provided stride — that's mechanism (b), not a clean contract extension. |
| **(b)** new tiling mode "against a strided destination" | **REJECT (primary).** Viable but maximal SCF churn: a new public entry point + a new param on `generateLoopNestUsingForallOp:556`, plus the caller (IREE) must extract+pass the stride — duplicating slice-op geometry. It is a special-case "strided destination mode" rather than a natural consequence of an op's own result position. Keep as a noted alternative (§11). |
| **(c)** give `insert_slice`/`parallel_insert_slice` a `TilingInterface` impl declaring a strided result position | **CHOSEN.** The op whose *result* is the strided destination declares its own strided `getResultTilePosition`. Stride source = the insert_slice's `strides` attribute (`MixedStrides`). Flows cleanly through the existing initial-tiling call chain (§1.3). Minimal SCF change (contract out-param + read channel at `:616-617`). Production mechanism, not a synthetic stand-in. |
| **(d)** test-only strided op | **REJECT for the probe.** Would prove "SCF can emit a stride" but NOT "insert_slice can be the strided anchor" (the production op). The stride genuinely lives on insert_slice; a stand-in leaves the real topology unproven. (The refuted plan dropped `Test_TilingNoDpsOp` for the same reason; that judgment holds.) |

**Why (c) is initial tiling, not the refuted consumer fusion:** the insert_slice
becomes the **anchor** tiled by `tileUsingSCF`/`tile_using_forall`. Its result
tensor IS the strided destination, so its `getResultTilePosition` legitimately
reports a strided position. This is the *opposite* direction from consumer
fusion (which fuses a strided slice *into* a tiled loop and can only preserve
the unit-stride candidate). There is no circularity.

### 1.3 The exact call chain — strided destination → `:616-617` writeback

All line numbers verified in this checkout. The driver is
`transform.structured.tile_using_forall` (or `tile_using_for`) on the
`tensor.insert_slice` anchor:

```
transform.structured.tile_using_forall  (%r = tensor.insert_slice … [0,0][2,4][2,1])   // sizes == source shape (F2)
  └─ mlir::scf::tileUsingSCF(rewriter, op=insert_slice, options)   [TileUsingInterface.cpp:1112]
       │  iterationDomain = op.getIterationDomain()                [impl: insert_slice source shape, e.g. [2,4]]
       │  innerYieldTiledValuesFn = GenerateTiledBodyFn lambda     [:1158-1226]
       │  initTensors = createInitialTensorsForTiling(...)         [:1229] → getOrCreateDestinations → [%dest]  [:734]
       └─ generateLoopNest(..., initTensors=[%dest], innerYieldTiledValuesFn)   [:1241]
            └─ generateLoopNestUsingForallOp(..., destinationTensors=[%dest], tiledBodyFn)   [:713]
                 │  getTileOffsetAndSizesWithForAllOp(ivs, loopRanges, givenTileSizes, numThreads)   [:602]
                 │      → offsets = [iv*… , 0], sizes = [tileSize, 4]   (iter-domain tile of the SOURCE)   [:473-544]
                 │  tiledBodyFn(rewriter, loc, ivs, offsets, sizes, innerDest=[%destRegionArg],
                 │              tiledResults, resultOffsets, resultSizes)   [:607]
                 │     // (5c) tile the anchor:
                 │     getTiledImplementation(insert_slice, offsets, sizes)   [:1194]
                 │         → %src_tile = tensor.extract_slice %filled[offsets][sizes][1,1]   (contiguous source tile)
                 │     // (5e) result position — THE CONTRACT SURFACE:
                 │     getResultTilePosition(insert_slice, 0, offsets, sizes,
                 │         resultOffsets, resultSizes, /*NEW*/ resultStrides)   [:1211-1214]
                 │         impl returns:
                 │           resultOffsets = [insert_base[0] + offsets[0]*strides[0], …] = [iv*2, 0]
                 │           resultSizes   = iterSizes (source tile shape)         = [1, 4]   (§3, R1 F1)
                 │           resultStrides = strides                                         = [2,1]   ← STRIDE ENTERS HERE
                 │     tiledResults = [%src_tile]
                 │  for each (tiledValue, dest, resultOffset, resultSize /*NEW: resultStride*/):
                 │     tensor::ParallelInsertSliceOp::create(…, %src_tile, %destRegionArg,
                 │         [iv*2,0], [1,4], resultStride)            [:616-621]   ← STRIDED WRITEBACK EMITTED
                 └─ return forall
```

**The stride reaches `:616-617` through exactly three hops:**
1. `getResultTilePosition` returns it (new `resultStrides` out-param) — `:1211`.
2. `innerYieldTiledValuesFn` (the `GenerateTiledBodyFn`) carries it out (new
   `resultStrides` field on the typedef, `:359-364`) — `:1221-1222` region.
3. `generateLoopNestUsingForallOp:616-617` reads it instead of
   `rewriter.getIndexAttr(1)`.

**The insert_slice `getResultTilePosition` math (static-2, identity write):**
- `resultOffsets[d] = insert.getMixedOffsets()[d] + iterOffsets[d] * insert.getMixedStrides()[d]`
- `resultSizes[d]   = iterSizes[d]`   (the SOURCE tile shape; NOT iterSizes*stride — the dest span (size-1)*stride+1 is implicit and checked in-bounds. Authority: `verifyInsertSliceOp` `TensorOps.cpp:2885-2896`, `inferResultType(dst, sizes)` is sizes-only. R1 F1.)
- `resultStrides[d] = insert.getMixedStrides()[d]`

For tile-size-1, stride-2, base `[0,0]`, iter `[iv,0]/[1,4]`:
`offset=[iv*2, 0]`, `size=[1, 4]` (= iterSizes), `stride=[2,1]`. ✓ (matches the expert-confirmed `iv*2`; this tile-size-1 example is the offset-placement path — see §3 for why a size-1 strided dim is a VESTIGIAL stride and the genuine-scatter proof needs the full-source case).

> **Note on the source tile & writeback dest:** `getTiledImplementation` for
> insert_slice emits `tensor.extract_slice %source[off][sz][1…]` (the source is
> contiguous; the stride lives entirely in the *writeback*, never in the read).
> The per-tile `parallel_insert_slice` dest is the loop's **region iter arg**
> (`forallOp.getRegionOutArgs()`, `TileUsingInterface.cpp:597`) — the
> `shared_out` block argument, **NOT** the outer `%dest` value (the outer
> `%dest` becomes the forall `shared_outs` at `:585-592`; the writeback at
> `:619-621` zips over `innerDestinationTensors` = region iter args, `:613-614`).
> The producer `linalg.fill` is fused in only for the EXEC/realism stretch
> (§5-G4); the core capability does not need it.

---

## 2. Contract-change scope (precise edits)

### 2.1 `TilingInterface.td` — ONE new out-param, in scope

`include/mlir/Interfaces/TilingInterface.td:118-162` — `getResultTilePosition`.
Add a 7th argument and document it:

```tablegen
InterfaceMethod<
  /*desc=*/[{ … (existing) …
    - `resultStrides` is the stride of the tile of the result generated
      by the tiled implementation, in the coordinate system of the
      result. A unit stride (`1`) on every dimension is the default and
      describes a contiguous result tile; a non-unit stride describes a
      dilated/scattered tile (e.g. a strided `tensor.insert_slice`
      writeback). The surrounding loop construct must emit the writeback
      with these strides. Implementations that cannot express a stride
      for a given result should set every dimension to `1`.
  }],
  /*retType=*/"::llvm::LogicalResult",
  /*methodName=*/"getResultTilePosition",
  /*args=*/(ins
    "::mlir::OpBuilder &":$b, "unsigned":$resultNumber,
    "::mlir::ArrayRef<::mlir::OpFoldResult> ":$offsets,
    "::mlir::ArrayRef<::mlir::OpFoldResult> ":$sizes,
    "::mlir::SmallVector<::mlir::OpFoldResult> &":$resultOffsets,
    "::mlir::SmallVector<::mlir::OpFoldResult> &":$resultSizes,
    "::mlir::SmallVector<::mlir::OpFoldResult> &":$resultStrides),   // NEW
  /*defaultImplementation=*/[{
    // Default: contiguous tile. Existing implementors are unaffected.
    resultStrides.assign(resultSizes.size(),
                         ::mlir::OpBuilder(::mlir::OpBuilder::ArgumentList{}).getIndexAttr(1));
    return failure();
  }]
>;
```

> Default fills unit strides so **every existing implementor is unaffected**
> (they never populate the new out-param; the default fires). The probe only
> needs the *anchor* to populate it.

**Operand-tile methods (`:202-301`) — explicitly OUT of scope.**
`getTiledImplementationFromOperandTiles` / `getIterationDomainTileFromOperandTiles`
serve the "tile producer, fuse consumer" direction. The probe uses **initial
tiling** (`getTiledImplementation` + `getResultTilePosition`), so these are not
on the critical path. Adding strides to them is a consumer-fusion-direction
follow-on (the refuted plan's territory); deferred (§9).

### 2.2 Implementors — who populates `resultStrides`

| Implementor | File:anchor | What it returns for static-2 |
|---|---|---|
| **NEW `tensor.insert_slice`** | new model in `lib/Dialect/Tensor/IR/TensorTilingInterfaceImpl.cpp`, registered at `:311-316` next to `PadOp` | `resultStrides = getMixedStrides()` → `[2,1]` (Case-1) / `[1,2]` (transpose). `resultOffsets/Sizes` per §1.3 math. **The only non-unit implementor.** |
| linalg (fill, generic, …) | `TilingInterfaceImpl.cpp:235-259` | unchanged body + `resultStrides.assign(rank, b.getIndexAttr(1))` (unit; projected-permutation maps cannot stride) |
| `tensor.pad` | `TensorTilingInterfaceImpl.cpp:57-66` | unit (pad result is contiguous) |
| `tensor.pack` / `tensor.unpack` | `TilingInterfaceImpl.cpp:1029`, `:1499` | unit (non-trivial offset/size map, but contiguous result) |

The four existing implementors gain one line each (assign unit strides). The
`insert_slice` model is the new ~60-90-line implementor (modeled on the `PadOp`
model at `:24-84` + `OffsetSizeAndStrideOpInterface` accessors).

### 2.3 SCF internal channels — two typedefs, six sites

- `GenerateTiledBodyFn` (`TileUsingInterface.cpp:359-364`): add
  `SmallVector<SmallVector<OpFoldResult>> &resultStrides`. Drives the
  **initial-tiling** writeback pair: `:447-448` (`scf.for`) and **`:616-617`
  (`scf.forall`)** — the probe's critical sites.
- `YieldTiledValuesFn` (`:336-340`): add the same field. Drives the
  **fusion** writeback pair `:951-955` (`scf.for`) / `:1006-1007` (`scf.forall`)
  and the dest-extract pair `:1565-1566` (producer) / `:2366-2367` (consumer).
- All six sites: replace `SmallVector<OpFoldResult> resultStride(N,
  rewriter.getIndexAttr(1))` with `resultStrides[i]` (the channel value),
  defaulting to unit everywhere except the insert_slice anchor path.
- Static helper `getResultTilePosition` (`:848-856`): add a `resultStrides`
  pass-through param so the lambda at `:1211-1214` can forward it.

---

## 3. The canonical static case — sub-cases A (genuine scatter) and B (offset placement)

> **Why two sub-cases — the vestigial-stride trap (verified vs the verifier).**
> `OffsetSizeAndStrideOpInterface` semantics: a `parallel_insert_slice
> %tile into %dest[off][sizes][strides]` places source element $j$ at dest index
> $\text{offset} + j\cdot\text{stride}$ (`TensorOps.cpp:3960-3982` verifier →
> `verifyInsertSliceOp` `:2885-2896`). For a strided dim where $\text{size}=1$,
> $j$ takes only the value $0$, so placement $=\text{offset}+0=\text{offset}$ —
> **identical to unit stride**. The verifier checks `sizes` against the source
> shape (`inferResultType(dst, sizes)` is sizes-only) and in-bounds; it does NOT
> check that the stride is "meaningful." So a size-1 strided dim is a *vestigial*
> stride: a hardcoded `[1,1]` writeback passes it byte-for-byte. **A green on a
> size-1 strided dim proves only "SCF computed the strided OFFSET," not "SCF
> emits a genuine strided scatter."** The capability proof therefore requires a
> strided dim with $\text{size}\ge 2$.

### Sub-case A (PRIMARY G3/G4) — full-source 1-tile genuine scatter

**This is the load-bearing probe.** The whole source is one tile, so the
stride is exercised *within the tile* (not just in the per-tile offset).

Input IR (anchor = the `tensor.insert_slice`; **F2: sizes = source shape**):
```mlir
func.func @strided_write_2x(%dest: tensor<4x4xi32>) -> tensor<4x4xi32> {
  %c1 = arith.constant 1 : i32
  %init = tensor.empty() : tensor<2x4xi32>
  %filled = linalg.fill ins(%c1 : i32) outs(%init : tensor<2x4xi32>) -> tensor<2x4xi32>
  // source(2,4); each source row j lands at dest row 0+j*2 -> dest rows {0,2}
  %r = tensor.insert_slice %filled into %dest[0, 0] [2, 4] [2, 1]
        : tensor<2x4xi32> into tensor<4x4xi32>
  return %r : tensor<4x4xi32>
}
```
Verify the anchor against the verifier by hand: `inferResultType(dest<4x4>,
[2,4])` = `tensor<2x4>` == source ✓ (`TensorOps.cpp:2891-2895`); in-bounds span
dim-0 = `0 + (2-1)*2 = 2 < 4` ✓, dim-1 = `0 + (4-1)*1 = 3 < 4` ✓ (`:2910-2912`).

After `transform.structured.tile_using_forall` on `%r` with `tile_sizes [2,4]`
(= full source → 1 iteration), the expected per-tile writeback (`iv = 0` only):
```mlir
// inside scf.forall (1 iteration); %o0 = the forall's region iter arg (shared_out)
%tile = tensor.extract_slice %filled[0, 0] [2, 4] [1, 1]  // contiguous source tile (the whole source)
  : tensor<2x4xi32> to tensor<2x4xi32>
tensor.parallel_insert_slice %tile into %o0[0, 0] [2, 4] [2, 1]
  : tensor<2x4xi32> into tensor<4x4xi32>
```
- **Why this is load-bearing (the false-green guard):** tile row 0 → dest row 0,
  tile row 1 → dest row `0 + 1*2 = 2`. A hardcoded `[1,1]` writeback would place
  tile row 1 at dest row **1** (wrong). So this green is achievable ONLY if the
  channel threads a genuine `[2,1]` stride from `getResultTilePosition` through
  to `:616-617`. This is what makes it the capability proof.
- **G3 IR FileCheck (sub-case A, fold-robust per m3):** the load-bearing tokens
  are the `[2,4]` sizes (== source) AND `[2,1]` strides *together*. A 1-iteration
  `scf.forall` is a canonicalization target: if it folds, the writeback surfaces
  as a plain `tensor.insert_slice` (terminator rewrite) rather than
  `tensor.parallel_insert_slice`, and the dest may switch from the region iter arg
  `%o0` to the folded target — so a FileCheck anchored on the op name or on `%o0`
  would be a **spurious RED under folding, not a mechanism failure**. Match the
  **geometry under EITHER op mnemonic**, e.g.
  `// CHECK: tensor.{{(parallel_)?}}insert_slice {{.*}}[0, 0] [2, 4] [2, 1]`. The
  genuine-stride invariant survives the fold (the `[2,1]` scatter is not
  copy-elidable), so this geometry line is stable. **Fallbacks** if even the
  geometry regex misses (residual rewrite risk): (i) drop `-canonicalize` from the
  IR-check `mlir-opt` run, or (ii) use the 2-tile genuine scatter
  `source<4x4> → dest<8x4>`, `tile_sizes [2,4]` → 2 non-foldable tiles, each a
  genuine scatter (the G4b shape, §7). G4 (EXEC) is unaffected — it consumes the
  pre-canonicalization IR through the full lowering pipeline (§8).

### Sub-case B (SECONDARY) — tile-size-1 strided-offset placement

Same input, but `tile_sizes [1,4]` (2 iterations). The expected per-tile
writeback (`iv ∈ {0,1}`):
```mlir
%off0 = arith.muli %iv, %c2 : index                         // iv * stride(2)
%tile = tensor.extract_slice %filled[%iv, 0] [1, 4] [1, 1]  // contiguous source tile (1 row)
  : tensor<2x4xi32> to tensor<1x4xi32>
tensor.parallel_insert_slice %tile into %o0[%off0, 0] [1, 4] [2, 1]
  : tensor<1x4xi32> into tensor<4x4xi32>
```
- **What B proves:** the strided *offset* composition `iv*2` (`arith.muli %iv,
  %c2`) — that `getResultTilePosition`'s `offset[d] = base[d] +
  iter_offset[d]*str[d]` is wired. (`resultSizes = iterSizes = [1,4]` per F1.)
- **Why B is a FALSE-GREEN alone:** dim-0 size = 1, so the `[2,1]` stride is
  vestigial (see the trap above). A hardcoded `[1,1]` writeback passes B
  identically — both write dest rows {0,2} via the offset alone. **B cannot
  prove the capability; it only proves offset placement.** G3/G4 green must be
  earned on sub-case A.

### No-overlap contract — RESOLVED under `resultSizes = iterSizes`

Under the corrected convention, tile `iv` writes the dest point-set
$\{\text{base}+(\text{iter\_offset}(iv)+j)\cdot\text{str} : j\in[0,\text{iterSize})\}$.
Adjacent tiles satisfy $\text{iter\_offset}(iv{+}1)=\text{iter\_offset}(iv)+\text{iterSize}$,
so the gap between tile $iv$'s last element and tile $iv{+}1$'s first is exactly
$\text{str}\ge 1$ — **point-set disjoint for all stride ≥ 1, tile ≥ 1**
(`TilingInterface.td:146-147` satisfied). The only contract-violating variant
was the discarded `*stride` "span" convention (A1's F1 bug); it is gone.

### R2 enforcement (sub-case A is the pin)

Every GREEN FileCheck asserts the COMPLETE per-tile IR. Sub-case A's check pins
`[2,4]` (sizes == source) AND `[2,1]` (genuine stride) together; sub-case B's
pins `arith.muli %iv, %c2` AND `[1,4]`. Matching a stride attribute while the
sizes are wrong (the discarded span convention) would PASS on paper and
miscompile in silicon — R2 exists to forbid that.

**Expected EXEC result (sub-case A and B agree):** over an all-zero `%dest` →
rows `{0,2}` become `1`, rows `{1,3}` stay `0`:
```
1 1 1 1
0 0 0 0
1 1 1 1
0 0 0 0
```

**Case 1′ (stretch) — dim-1 strided transpose.** `insert_slice
%src<4x2> into %dest<4x4>[0,0][4,2][1,2]` (sizes = source shape), `tile_sizes
[4,2]` → 1-tile genuine scatter on dim-1 (col 1 → dest col 2). Cheap
per-dimension coverage; keep it a genuine-scatter case (size ≥ 2 on the strided
dim), not a vestigial one.

---

## 4. Inventory (re-verified this session — all line numbers current)

| # | Site | Kind | On probe critical path? | Action |
|---|---|---|---|---|
| 1 | `TileUsingInterface.cpp:447-448` | initial `scf.for` writeback hardcode | YES (for-op twin) | read channel |
| 2 | `TileUsingInterface.cpp:616-617` | initial `scf.forall` writeback hardcode | **YES (primary)** | read channel |
| 3 | `TileUsingInterface.cpp:951-955` | fusion `scf.for` writeback hardcode | no (unit under probe) | read channel |
| 4 | `TileUsingInterface.cpp:1006-1007` | fusion `scf.forall` writeback hardcode | no (unit under probe) | read channel |
| 5 | `TileUsingInterface.cpp:1565-1566` | producer dest-extract hardcode | only Case-2-stretch | read channel |
| 6 | `TileUsingInterface.cpp:2366-2367` | consumer dest-extract hardcode | no (consumer fusion, refuted direction) | read channel |
| R1 | `TileUsingInterface.cpp:2313-2317` | consumer candidate rejection | no | leave (consumer-fusion follow-on) |
| R2 | `TileUsingInterface.cpp:1502-1504` | producer candidate rejection | only Case-2-stretch | leave until G5 |
| G1 | `SwapExtractSliceWithProducerPatterns.cpp:31-33` | tensor extract-swap stride guard | no (transform producer fusion) | flip in lockstep (§6) |
| G2 | `SwapExtractSliceWithProducerPatterns.cpp:99-101` | tensor insert-swap stride guard | no (consumer fusion util) | flip in lockstep (§6) |
| — | `candidateSliceOp.getMixedStrides()` read | `TileUsingInterface.cpp:2311` | no (consumer path) | n/a |
| — | `getResultTilePosition` interface | `TilingInterface.td:118-162` | **YES** | add `resultStrides` |

**File-path correction** (the expert review and prior docs cite this as
`Linalg/Transforms/`): `SwapExtractSliceWithProducerPatterns.cpp` actually lives
at `third_party/llvm-project/mlir/lib/Dialect/Tensor/Transforms/` (verified by
glob). Both guards carry the smoking-gun comment
*"`TilingInterface` currently only supports strides being 1."* (`:31`, `:99`).

## 4a. Blast-radius containment (F3) — gating the new interface's *use*

Giving `tensor.insert_slice` `TilingInterface` is a **global** change: ~30
`dyn_cast<TilingInterface>` / `isa<TilingInterface>` sites in MLIR + IREE flip
from "skip" to "match." The dangerous ones (verified this session) are in the
**real bug's pipeline**:

- **`GPUGreedilyDistributeToThreads.cpp:139`** —
  `if (auto tilableOp = dyn_cast<TilingInterface>(op))` inside an IR **walk**
  (`processRegion`, `:114-154`); any `insert_slice` it meets is routed to
  `tileToThreads` (`:145`). This is in the GPU distribution path the
  `m[0::2,0::2]=True` dispatch traverses.
- **`TileAndFuseUtils.cpp`** — producer/consumer fusion worklists keyed on
  `TilingInterface`: `:40` (`getDefiningOp<TilingInterface>`),
  `:78`/`:90` (`isa<TilingInterface>` on producer/user), `:271`/`:394`
  (`dyn_cast<TilingInterface>` on consumer/owner). An `insert_slice` becomes a
  fusable producer/consumer.
- **`TileUsingInterface.cpp:2027`** — consumer-fusion gate that ALSO requires
  `isa<DestinationStyleOpInterface>`; `insert_slice` *is* DestinationStyle
  (`TensorOps.td:843`), so it passes BOTH gates and becomes fusion-eligible.
  (R1 §4 lists the fuller set: `GPUFuseAndHoistParallelLoops`, `GPUTensorTile`,
  `GPUConvertToCoalescedDMA`, `GPUTile`, `CPUPrepareUkernels`, etc.)

This does NOT block G3/G4 (the probe is a pure transform-interpreter test that
targets a single explicit payload op), but the GO criterion authorizes
proceeding toward that integration, so containment must be specified now.

**Containment — two firewalls (R1 §4, adopted):**

1. **Marker-gated impl (PRIMARY firewall, gates the interface's *use* not its
   *registration*).** The marker-gate lives on the two impl methods that can
   express failure — **`getTiledImplementation`** (returns
   `FailureOr<TilingResult>`, `TilingInterface.td:107`) and
   **`getResultTilePosition`** (returns `LogicalResult`, `:149`). It does **NOT**
   live on `getIterationDomain`: that method returns `SmallVector<Range>`
   (`:80-85`, default `return {}`) — there is no `LogicalResult` to return, so it
   reports the **true source-shape domain unconditionally**. Both gated methods
   early-return failure unless a discardable marker attribute (e.g.
   `transform.marker` / an IREE lowering-config flag, set by the transform anchor
   on its target payload op) is present. **Verified bail path** (opened this
   session): a greedy distributor's `dyn_cast<TilingInterface>` now *succeeds* on
   `insert_slice` (`GPUGreedilyDistributeToThreads.cpp:139`, routed to
   `tileToThreads` `:145` → `tileConsumerAndFuseProducersUsingSCF` `:93-95` →
   `tileUsingSCF`), which calls `getIterationDomain` (valid ranges, ungated) and
   builds the forall; the body lambda then calls the static helper
   `getTiledImplementation` (`TileUsingInterface.cpp:820-829`, forwarding to the
   interface method `op.getTiledImplementation`) — on an *unmarked* op it returns
   failure, the lambda erases its clone and returns `op.emitOpError("failed to
   tile operation")` (`:1197-1199`); `generateLoopNestUsingForallOp:607-610`
   converts that into `notifyMatchFailure("failed to generate loop body")`;
   `tileUsingSCF:1244-1245` fails ("failed to generate tiling loops");
   `tileToThreads` bails silently (`GPUGreedilyDistributeToThreads.cpp:96-98`).
   **No strided writeback is ever emitted** — the `parallel_insert_slice` at
   `:619-621` is downstream of the failing call and is never reached. Only the
   explicit transform probe (which sets the marker) clears both gates and
   exercises the strided path. Additionally, the impl **deliberately omits** the
   operand-tile/fusion methods (`generateResultTileValue`,
   `getTiledImplementationFromOperandTiles`; PadOp has them at `:79-83`, this impl
   does NOT), so fusion consumers (`TileUsingInterface.cpp:2027`,
   `SwapExtractSliceWithProducerPatterns`) get the default `failure()` and are
   safe by construction.
2. **IREE anchor allow-list (DEFENSE-IN-DEPTH, integration PR).** IREE's anchor
   selection (`TileDispatchUsingForall.cpp:67-76`; `isComputeOp` =
   `TilingInterface | UKernelOpInterface`, `Utils.cpp:980-982`) and the
   `TileAndFuseUtils` worklists must filter on an explicit allow-list (not the
   bare `TilingInterface` cast) before admitting `insert_slice` as a
   distributable/fusable op. This is the real integration firewall — out of
   scope here (§9) but its contract is stated now so the marker gate (1) and the
   allow-list (2) compose.

**Trade-off (acknowledged):** a guarded production impl "smells like a test-only
op" — the reason A1 rejected mechanism (d). But a guarded *real* impl (real
geometry, real stride threading via the verified `:616` channel) gated for
staged rollout is materially different from a synthetic stand-in; it is the
correct firewall shape for enabling a global interface change behind a flag.

**Containment limits — best-effort, not hard (m4, honest):** firewall 1 is a
*behavioral* gate, not a type-system guarantee; it does not, on its own,
hard-contain the global interface change. What it does NOT catch:
- **A partial/empty `scf.forall` body may be left in the IR on an unmarked hit.**
  The gate fires *inside* the body lambda (`TileUsingInterface.cpp:607`), which
  runs *after* the forall shell is already created (`scf::ForallOp::create`
  `:585`); `notifyMatchFailure` (`:610`) logs the bail but does **not** erase that
  forall. Whether the caller rolls it back is an **execution checkpoint, not a
  guarantee** (R2 risk #4) — the integration PR must confirm or force the
  rollback, or a stray empty forall could surface in the real GPU pipeline. (It
  carries no strided writeback, so it is not a *correctness* leak — only a
  noise/regression leak.)
- **A consumer that bypasses the gated methods.** Any pass that builds a tiled
  body for `insert_slice` *without* routing through
  `getTiledImplementation`/`getResultTilePosition` — e.g. a future transform
  driving the operand-tile methods, or one that constructs a `scf.forall` +
  `insert_slice` directly — is not contained by the marker. The omitted-methods
  default (`failure()`) covers *today's* fusion paths only.
- **`tileToThreads`'s silent bail is a property of *current* code**
  (`GPUGreedilyDistributeToThreads.cpp:96-98`, "returns silently"). If a future
  change made it hard-error instead of bail, or stopped routing through
  `getTiledImplementation`, an unmarked `insert_slice` would be mis-tiled.
The **hard** firewall is therefore firewall 2 (the IREE allow-list), which MUST
ship in the integration PR before any marker is dropped. The marker gate is a
staged-rollout brake, not a load-bearing containment boundary on its own.

---

## 5. Staged RED/GREEN plan (propagate-first, flip-last; static stride 2)

**Invariants (gate EVERY commit):**
- **R1 — propagate-first, flip-last.** No rejection guard (R1/R2/G1/G2) is
  removed before the stride is wired through *all* sites that guard feeds. The
  channel is threaded and defaults to unit *first*; flips are the last atomic
  action of the stage that needs them.
- **R2** — every GREEN FileCheck asserts the COMPLETE per-tile IR (§3).
- **R3** — the go/no-go stage culminates in a `mlir-cpu-runner` EXEC.
- **R4** — a unit-stride twin (`[1,1]`) of every strided test stays green at
  every commit.

> Build once at Stage 0: `cmake --build /Users/alex/Developer/.iree-build
> --target mlir-opt FileCheck llvm-lit mlir-cpu-runner`. MLIR lit suite is NOT
> configured → use direct `mlir-opt | FileCheck` (config-independent).

### Stage 0 — harness baseline (no behavior change)
- [ ] **0.1** Build the four targets (above).
- [ ] **0.2** Smoke a known-good tiling test:
  `mlir-opt --transform-interpreter --split-input-file -canonicalize
  third_party/llvm-project/mlir/test/Dialect/Linalg/transform-op-fuse.mlir |
  FileCheck …transform-op-fuse.mlir` → exit 0. Stop if not green.
- [ ] **0.3** Record today's RED: run the Stage-1 test through
  `tile_using_forall` on the insert_slice → today it fails at
  `tileToForallOpImpl`'s `dyn_cast<TilingInterface>(target)`
  (`LinalgTransformOps.cpp:3899-3900`, reached from `TileUsingForallOp::apply:3973`)
  with a silenceable "not TilingInterface" error, because `insert_slice` is not
  registered (`TensorTilingInterfaceImpl.cpp:314` registers only `PadOp`).
  Capture the error.

### Stage 1 — RED (failing tests first)
- [ ] **1.1** Create
  `third_party/llvm-project/mlir/test/Dialect/Tensor/tiling-insert-slice-strided.mlir`
  with the §3 anchor IR
  (`insert_slice %filled<2x4> into %dest<4x4>[0,0][2,4][2,1]`) and TWO transform
  sequences — sub-case A and sub-case B:
  ```mlir
  // sub-case A (PRIMARY): tile_sizes [2,4] -> 1-tile genuine scatter
  %la, %ta = transform.structured.tile_using_forall %m tile_sizes [2, 4] ...
  // sub-case B (offset placement): tile_sizes [1,4] -> 2-tile, vestigial stride
  %lb, %tb = transform.structured.tile_using_forall %m tile_sizes [1, 4] ...
  ```
  (mapping attrs optional; drop if they complicate the test.) Sub-case A's
  FileCheck asserts the genuine scatter
  `parallel_insert_slice %tile into %o0[0,0][2,4][2,1]`; sub-case B's asserts
  `arith.muli %iv, %c2` + `[1,4][2,1]`.
- [ ] **1.2** Run:
  `.iree-build/llvm-project/bin/mlir-opt --transform-interpreter <test>.mlir |
  .iree-build/llvm-project/bin/FileCheck <test>.mlir`. Observe **FAIL** on BOTH
  — `tileToForallOpImpl` rejects the non-`TilingInterface` target
  (`LinalgTransformOps.cpp:3899-3900`). This is the RED. **R4 twins** (the
  `[1,1]`-stride version of each) are also RED here (same cause) — they go
  green in G3 and must stay green after.
- [ ] **1.3** Write the EXEC harness (§8) asserting the §3 result matrix
  (rows {0,2} = 1); it cannot run yet (no strided IR). RED recorded.

### Stage 2 — GREEN-propagate (contract + channel; ZERO behavior change)
**This commit changes no observable behavior — every default is unit.** Its sole
purpose is to thread the channel so later stages can populate it. Existing
tiling tests MUST stay green (R4).

- [ ] **2.1** `TilingInterface.td:118-162`: add `resultStrides` out-param +
  unit default (§2.1). Regenerate the interface header.
- [ ] **2.2** Populate unit strides in the four existing implementors
  (linalg `:235-259`, pad `:57-66`, pack `:1029`, unpack `:1499`) — one
  `resultStrides.assign(rank, b.getIndexAttr(1))` line each.
- [ ] **2.3** `TileUsingInterface.cpp`: add `resultStrides` to
  `GenerateTiledBodyFn` (`:359-364`) and `YieldTiledValuesFn` (`:336-340`);
  thread through the static helper `getResultTilePosition` (`:848-856`) and the
  two body lambdas (`:1158-1226` initial; `:2298` consumer).
- [ ] **2.4** At all six sites (`:447-448`, `:616-617`, `:951-955`,
  `:1006-1007`, `:1565-1566`, `:2366-2367`) read the channel value in place of
  the hardcoded `getIndexAttr(1)`. The channel is unit everywhere, so the
  emitted IR is byte-identical to today.
- [ ] **2.5** Build. Re-run Stage-0.2 smoke + the R4 unit-stride tiling suite
  (transform-op-fuse, transform-op-tile, …). All must stay green.
  Commit: `mlir: thread resultStrides through TilingInterface + SCF channels (default unit, no behavior change)`.

### Stage 3 — GREEN-anchor (insert_slice impl; the go/no-go IR)
- [ ] **3.1** In `TensorTilingInterfaceImpl.cpp`, add an
  `InsertSliceOpTiling : public TilingInterface::ExternalModel<…, InsertSliceOp>`
  modeled on `PadOpTiling` (`:24-84`):
  - `getLoopIteratorTypes`: `parallel` × source-rank.
  - `getIterationDomain`: the **source** tensor's shape (offset 0, size =
    source dims, stride 1) — reify from the source type.
  - `getTiledImplementation(b, offsets, sizes)`: emit
    `tensor.extract_slice %source[offsets][sizes][1…]` (contiguous source tile);
    return it as the single tiled value. (No strided read — the stride is in the
    writeback only.)
  - `getResultTilePosition(b, 0, offsets, sizes, rOff, rSize, rStrides)`: the
    §1.3 math — `rOff[d]=base[d]+offsets[d]*str[d]`,
    `rSize[d]=sizes[d]` (= iterSizes, the SOURCE tile shape; NOT `sizes*stride`
    — R1 F1, `verifyInsertSliceOp` is sizes-only), `rStrides[d]=str[d]`, where
    `base/str = getMixedOffsets()/getMixedStrides()`. For sub-case A
    (tile_sizes [2,4]): `rOff=[0,0]`, `rSize=[2,4]`, `rStrides=[2,1]`; for
    sub-case B (tile_sizes [1,4]): `rOff=[iv*2,0]`, `rSize=[1,4]`, `rStrides=[2,1]`.
- [ ] **3.2** Register at `:311-316`:
  `tensor::InsertSliceOp::attachInterface<InsertSliceOpTiling>(*ctx);`
- [ ] **3.3** Build. Run Stage-1.1: the Case-1 lit goes **GREEN** (strided
  writeback emitted at `:616-617`). R4 twin (`[1,1]`) stays green.
  Commit: `mlir/tensor: give tensor.insert_slice a TilingInterface impl with strided result position`.

> **GATE G3 (go/no-go, IR half):** initial tiling of a real slice op emits the
> **sub-case A** genuine-scatter writeback `parallel_insert_slice %tile<2x4>
> into %o0[0,0][2,4][2,1]` at `:616-617` (COMPLETE per-tile IR, R2). This is
> load-bearing: a hardcoded `[1,1]` would put tile row 1 at dest row 1, not row
> 2 — so green is achievable ONLY via the threaded `resultStrides` channel.
> Sub-case B (offset placement) must also green, but B-alone does NOT prove the
> capability (vestigial stride, §3). If the impl cannot produce A, stop.

### Stage 4 — EXEC (the go/no-go correctness half; G4a + G4b)
- [ ] **4.1 G4a — sub-case A EXEC (PRIMARY):** lower the sub-case-A tiled IR
  through the real pipeline (§8) and run `mlir-cpu-runner`. Assert rows `{0,2}`
  = 1 (the genuine-scatter result). This is the primary EXEC gate.
- [ ] **4.2 G4b — numThreads EXEC (PROMOTED from stretch; §7):** drive a
  **genuine-scatter multi-tile** case (`source<4x4> → dest<8x4>`, stride
  `[2,1]`, `num_threads [2]` → `ceilDiv(4,2)=2` per-tile source size, 2 tiles,
  EACH a genuine within-tile scatter of size-2 on the strided dim) through the
  same pipeline. Assert rows `{0,2,4,6}` = 1. This is the real-IREE-dispatch
  offset path (§7): it proves the strided offset composes correctly under
  `numThreads` AND that each per-tile writeback is a genuine scatter — neither a
  vestigial-stride nor a `tile_sizes`-only green can substitute for it.
  Commit (tests only): `mlir/test: EXEC for strided insert_slice initial tiling (tile_sizes + numThreads)`.

> **GATE G4 (go/no-go, the real gate):** G4a produces the correct
> genuine-scatter cells, G4b produces the correct `numThreads` cells, AND G3
> holds. This is the binary Phase-0 answer: *yes, the contract change lets the
> tiler emit a correct strided writeback for a real slice op, on BOTH the
> `tile_sizes` and the `numThreads` (real-IREE) offset paths.* No
> "fusion never fired" ambiguity.

### Stage 5 — stretches (de-risk more, in order)
- [ ] **5.1 Case 1′ (dim-1 transpose):** add the `[1,2]` lit + EXEC (genuine
  scatter on dim-1, size ≥ 2). Confirms per-dimension offset/size/stride
  composition. Cheap. (numThreads was Stage 5.2 in v1 — PROMOTED to core G4b.)
- [ ] **5.2 producer fusion:** use `tile_using_forall` + a producer-fusion
  transform so `linalg.fill` lands inside the loop (its source-tile is computed
  in-loop). Fill is contiguous → its writeback (`:1006`) is unit; no flip
  needed. Confirms the strided anchor coexists with unit-stride producer fusion.
- [ ] **5.3 tensor-dialect guards (§6):** flip G1/G2 in lockstep for systemic
  consistency. Verify no regression (they are not on the probe path).

### Stage 6 — regression sweep
- [ ] Run the representative tiling suites (R4 is per-commit; this is the wider net):
  `for t in transform-op-fuse transform-op-tile transform-op-tile-using-for
  tile-and-fuse-consumer; do mlir-opt --transform-interpreter --split-input-file
  …/$t.mlir | FileCheck …/$t.mlir; done`. Any regression → root-cause before
  declaring done.

---

## 6. The tensor-dialect guards — handled in lockstep (Stage 5.3)

Both at `lib/Dialect/Tensor/Transforms/SwapExtractSliceWithProducerPatterns.cpp`
(file path corrected from `Linalg/Transforms/` in prior docs):

- **G1 `:31-33`** (`replaceExtractSliceWithTiledProducer`): guards a strided
  `extract_slice` before calling `generateResultTileValue` (`:35-37`). After the
  contract change, `generateResultTileValue` for an insert_slice-as-producer
  source can return a strided tile; relax G1 to accept strides and forward them.
- **G2 `:99-101`** (`replaceInsertSlicesWithTiledConsumer`): the twin guard on
  candidate `insert_slice`s before
  `getTiledImplementationFromOperandTiles` (`:108-110`). Relax identically.

**Sequencing:** flip G1/G2 only after Stage-2 channel threading (so they have a
stride to forward) and only with a test that exercises the relaxed path. They
are **not on the G3/G4 critical path** (the probe uses initial tiling of
insert_slice, not these swap utilities), so a G1/G2 regression cannot spuriously
green or red the gate — but the contract change is inconsistent until they are
flipped, so they belong in the same PR set.

---

## 7. The `numThreads` path — PROMOTED to core G4b (§5 Stage 4.2)

IREE distributes via `numThreads`, not literal `tileSizes`
(`getTileOffsetAndSizesWithForAllOp:473-544`, `useNumThreads` branch `:574`).

> **PROMOTED to core G4b (orchestrator item 6a — verdict: JUSTIFIED).** IREE
> dispatches via `numThreads` on dynamic `tensor<?x?xi8>`, not literal
> `tile_sizes` on static tensors. A `tile_sizes`-only green (G4a) proves "SCF
> emits a strided writeback for a real slice op" but leaves the real-IREE offset
> path unproven: the `numThreads` branch computes per-tile offsets via
> *different code* (`getTileOffsetAndSizesWithForAllOp:473-544`, the
> `useNumThreads` branch at `:574`, offset `d0+d1*s0` at `:491`, plus
> residual/boundary handling `min/max` at `:514-538`) than the `tile_sizes`
> branch, so a `tile_sizes`-only green does NOT cover it. Promotion is
> **justified and cheap** — one extra lit + EXEC on the same harness — and it
> elevates the `iv*T*S` "by construction" math below from `[INFERENCE]` to
> evidence. It is promoted as a **genuine-scatter multi-tile** case (per-tile
> size ≥ 2 on the strided dim) so it is not itself a vestigial-stride false
> green. The dynamic-size half of the real dispatch (`tensor<?x?xi8>`) is NOT
> covered by this — it stays a separate, unproven boundary (§9, §10).

**Verified math:** with `numThreads`, the per-tile offset is
`loopRange.offset + iv * givenTileSize` (`offsetExpr = d0 + d1*s0`, `:491`),
where `givenTileSize` is the **caller-computed `ceilDiv(range, numThreads)`**
(the comment at `:525-528` confirms `ceilDiv(100,7)=15`). So
`iter_offset[d] = iv * ceilDiv(range, numThreads)` — already the *effective*
per-tile size.

The insert_slice `getResultTilePosition` receives these `offsets` (already
effective-size-adjusted) and computes `resultOffset[d] = base[d] +
offsets[d]*str[d]`. **The `iv*T*S` composition is therefore correct by
construction** — the impl never sees a "literal tileSize"; it sees the effective
`offsets` value. The assignment's concern ("`iv*T*S` assumes T is the true
per-tile size") is satisfied *for free* because T is baked into `offsets` by
`getTileOffsetAndSizesWithForAllOp` before `getResultTilePosition` runs.

**Test discipline (R2) — now G4b, not a stretch:** G4b must (i) use a case where
`ceilDiv(range,numThreads) != 1` so the EXEC distinguishes "effective tile used"
from "literal tile used," AND (ii) keep the per-tile strided dim **size ≥ 2**
(genuine scatter, not vestigial). The chosen case: `source<4x4> → dest<8x4>`,
stride `[2,1]`, `num_threads [2]` → `ceilDiv(4,2)=2` per-tile source size, 2
tiles; per-tile offset `iv*2*2 = iv*4`, each tile a size-2 genuine scatter → dest
rows `{0,2}` (tile 0) and `{4,6}` (tile 1). The Case-1 `range=2` probe would NOT
expose this (effective tile = 1, vestigial).

> **Dynamic-size gap (noted, out of scope):** the real dispatch is
> `tensor<?x?xi8>`; stride-dividing a *dynamic* span needs runtime `ceildivi`.
> The static probe does not de-risk this — the dynamic iteration domain under
> strided tiling is a separate question (§9).

---

## 8. EXEC gate — real pipeline from an Integration template

**Template captured from**
`mlir/test/Integration/Dialect/Vector/CPU/transfer-write.mlir:1-4` and the
ArmSVE `contraction.mlir` compile/run split (transform-interpreter → lower →
runner). Adapted for CPU:

```bash
# COMPILE: tile (transform) → bufferize → lower to LLVM. m2: the
# `-expand-strided-metadata -finalize-memref-to-llvm` pair is REQUIRED —
# `-finalize-memref-to-llvm` (NOT `-convert-memref-to-llvm`, which does not exist:
# Passes.td:994-995) owns `MemRefCopyOpLowering` (MemRefToLLVM.cpp:1140, :2109),
# the handler R2 §4 proved lowers the strided memref.copy (fork :1274-1277);
# omitting it → illegal `memref.copy`/`memref.subview` at LLVM conversion.
# Ordering per test/Integration/Dialect/Complex/CPU/correctness.mlir:1-11;
# strided-memref pair per test/Conversion/MemRefToLLVM/expand-then-convert-to-llvm.mlir:1.
# `-test-lower-to-llvm` (transfer-write.mlir:1) stays as monolithic fallback.
.iree-build/llvm-project/bin/mlir-opt %s \
  -transform-interpreter \
  -one-shot-bufferize="allow-return-allocs bufferize-function-boundaries" \
  -convert-linalg-to-loops \
  -convert-scf-to-cf \
  -expand-strided-metadata -finalize-memref-to-llvm \
  -convert-cf-to-llvm \
  -convert-arith-to-llvm -convert-math-to-llvm \
  -convert-func-to-llvm -convert-index-to-llvm \
  -reconcile-unrealized-casts \
  -o %t
```

# RUN:
.iree-build/llvm-project/bin/mlir-cpu-runner %t \
  -e entry -entry-point-result=void \
  -shared-libs=%mlir_c_runner_utils \
  | .iree-build/llvm-project/bin/FileCheck %s --check-prefix=EXEC
```
(`%mlir_c_runner_utils` resolves under the lit substitution model; for a direct
invocation use the absolute path
`.iree-build/llvm-project/lib/libmlir_c_runner_utils.dylib`. If
`mlir-cpu-runner` is unavailable, `mlir-runner` is the drop-in per
`transfer-write.mlir:2`.)

**ENTRY function** allocates a zero `memref<4x4xi32>` (G4a) or `memref<8x4xi32>`
(G4b), calls `@strided_write_2x` / `@strided_write_2x_nt` (via a wrapper that
materializes the tensor from the memref), and prints with `memref.print`. `EXEC`
checks the §3 matrix (G4a) / the §7 numThreads matrix (G4b).

**EXEC lowering path — TRACED (replaces the v1 `:672-674` claim, which cited the
WRONG op; R1 F4).** The G3 writeback is a `parallel_insert_slice` *inside*
`scf.forall`; it does **not** go through the standalone `tensor::InsertSliceOp`
bufferization A1 cited. The real path, opened this session:

1. **`scf.forall` bufferization** — `ForallOpInterface::bufferize`
   (`SCF/Transforms/BufferizableOpInterfaceImpl.cpp:1243-1296`): replaces each
   `shared_out` region arg with `to_tensor(memref)` and `mergeBlocks`
   (`:1260-1290`). It **does not inspect strides** — the terminator interfaces
   (`InParallelOp`/`ParallelInsertSliceOp`) are "only used during analysis. Not
   for bufferization" (`:1209-1212`). Stride survives this hop.
2. **`parallel_insert_slice` bufferization** — `ParallelInsertSliceOpInterface::bufferize`
   (`Tensor/Transforms/BufferizableOpInterfaceImpl.cpp:969-1027`, registered at
   `:1209-1210`): forwards `getMixedStrides()` **verbatim** into a
   `memref::SubViewOp` (`:998-1002`) + `options.createMemCpy` (`:1005-1006`).
   **There is no stride gate here.** The strided `memref.subview` + `memref.copy`
   is emitted faithfully. (The `allStridesOne` check A1 cited at `:672-674` lives
   on a *different* model — `InsertSliceOpInterface::bufferizesToMemoryRead` for
   the **standalone** `InsertSliceOp`'s in-place *analysis* decision `:655-675` —
   and is irrelevant to the `parallel_insert_slice` writeback path.)
3. **memref → LLVM** — `-expand-strided-metadata` + `-finalize-memref-to-llvm`
   (the pass owning `MemRefCopyOpLowering`, `MemRefToLLVM.cpp:1140`, registered
   `:2109`) lower the non-contiguous `memref.subview` + `memref.copy`. **The
   strided copy is NOT rejected** (resolves R2 §4, re-verified this session):
   `MemRefCopyOpLowering` forks on contiguity (`:1263-1277`) — a strided target
   fails the contiguous test and takes `lowerToMemCopyFunctionCall` (`:1277`), a
   generic element-wise runtime copy honoring both operands' layouts. So the
   *compile* path is not a wall; the residual is **performance** (scalarized
   strided copy, cf. §9 LLVM #51660), and cell correctness is proven ONLY by a
   green G4. The one COMPILE-fail mode is the pipeline **omitting**
   `-finalize-memref-to-llvm` (m2) → `memref.copy`/`memref.subview` as illegal ops.

**G4 risk, stated honestly:** bufferization (steps 1–2) preserves the stride by
construction — no gate rejects it. R2 §4 (re-verified: `MemRefToLLVM.cpp:1263-1277`)
shows the strided `memref.copy` is **not** rejected either — a strided target
fails the contiguous test and lowers to a correct, scalar element-wise runtime
copy (`lowerToMemCopyFunctionCall`, `:1277`). So the *compile* path is not a wall.
The real EXEC failure modes are: (a) the explicit pipeline **omits**
`-finalize-memref-to-llvm` (m2) → `memref.copy`/`memref.subview` survive as
illegal ops at LLVM-dialect conversion → COMPILE-fail, a distinct NO-GO mode
(§10) that still proves G3 (fix by adding the pass, or use `-test-lower-to-llvm`
monolithic fallback); (b) wrong cells → silent miscompile in offset/size/stride
composition. End-to-end cell correctness is proven ONLY by a green G4; the copy
is correct but **slow** (scalarized, cf. §9 LLVM #51660). **Validity of a strided
`parallel_insert_slice` inside `scf.forall` (verifier + bufferization survival)
is an explicit G4 acceptance criterion**, not a footnote: it verifies
(`TensorOps.cpp:3960-3982`) and its model forwards strides (`:969-1027`), both
confirmed by reading the code — but end-to-end survival is proven ONLY by a
green G4. (Upstreaming note, R1 §6 #7: the standalone `InsertSliceOp::bufferize`
author's own comment at `:694-698` flags tiled insert-slices as
"catastrophically bad scheduling" — it does NOT apply to the in-place
`parallel_insert_slice` form, but it signals where reviewer pushback on
upstreaming will land.)

---

## 9. Scope boundaries

**IN (this plan):**
- Vendored `third_party/llvm-project` MLIR only.
- `getResultTilePosition` contract: one new `resultStrides` out-param (unit
  default).
- `tensor.insert_slice` `TilingInterface` impl (the strided anchor).
- SCF `GenerateTiledBodyFn`/`YieldTiledValuesFn` channels + the six sites read
  from them.
- Static stride `2`, identity writeback (fill/insert_slice): sub-case A
  (genuine scatter, `<2x4>→<4x4>`, the PRIMARY) + G4b multi-tile
  (`<4x4>→<8x4>`); demoted sub-case B (tile-size-1 offset placement).
- The two tensor-dialect guards (G1/G2) flipped in lockstep.
- `numThreads` path (PROMOTED to core G4b — genuine-scatter multi-tile, §7).

**OUT (noted as gaps, not done here):**
- **Dynamic sizes** (`tensor<?x?xi8>`): the **REAL IREE dispatch shape**.
  Stride-dividing a *dynamic* span needs runtime `ceildivi` and a dynamic
  iteration domain; the static + `numThreads` probes do NOT transfer
  automatically. This is a **SEPARATE, UNPROVEN boundary** — Phase 0 (this
  plan) proves only the IR-level + static-EXEC capability; it does **NOT**
  claim `m[0::2,0::2]=True` compiles. (Orchestrator item 6b.)
- **Arbitrary / coprime / non-constant strides:** static `2` only.
- **Operand-tile contract methods** (`getTiledImplementationFromOperandTiles` /
  `getIterationDomainTileFromOperandTiles`): consumer-fusion direction; out.
- **Consumer-fusion rejection `:2313-2317`** and the consumer dest-extract
  `:2366-2367`: refuted direction for this op; revisit only if a future consumer
  of a strided slice is needed.
- **IREE filter widening** (`TileAndFuseUtils.cpp:141`, `:154-155`) +
  **anchor selection** (`TileDispatchUsingForall.cpp:67-76`, make insert_slice a
  computeOp with a lowering config) + **source-load co-distribution**: the real
  `m[0::2,0::2]=True` dispatch fix; pure consumer of this work, separate change.
  (Note: feasibility doc §4.3 #7's claim that "insert_slice *is* TilingInterface"
  is stale — §2.1 of the same doc corrects it; after this plan it becomes true.)
- **Standalone `insert_slice` in-place aliasing**
  (`InsertSliceOpInterface::bufferizesToMemoryRead` / `insertSliceOpRequiresRead`,
  `Tensor/Transforms/BufferizableOpInterfaceImpl.cpp:655-675`): the
  `allStridesOne` *analysis* gate — performance (the actual verifier-passing
  fix), not the compile path, AND **off the `parallel_insert_slice` writeback
  path entirely** (§8 F4 trace — that path is on a different model that forwards
  strides unconditionally).
- **LLVM #51660** (vector-dialect strided dense load/store): graceful
  performance degradation (vectorization disabled → scalar strided stores),
  correct but slow; not on the compile critical path.

---

## 10. Go/no-go gate (well-defined)

**GO iff ALL hold:**
- **G3 (IR, sub-case A PRIMARY):** initial tiling of a real `tensor.insert_slice`
  emits the **genuine-scatter** writeback `parallel_insert_slice %tile<2x4> into
  %o0[0,0][2,4][2,1]` at `:616-617` (COMPLETE per-tile IR, R2). Sub-case B
  (offset placement) also greens.
- **G4a (EXEC, `tile_sizes`, PRIMARY):** `mlir-cpu-runner` on the lowered
  sub-case-A IR writes rows `{0,2}` = 1, `{1,3}` = 0.
- **G4b (EXEC, `numThreads`, real-IREE offset path):** the genuine-scatter
  multi-tile case (`<4x4>→<8x4>`, `num_threads [2]`) writes rows `{0,2,4,6}` = 1.

**What GO proves — and what it does NOT (orchestrator item 6b):** GO proves the
**IR-level + static-EXEC capability**: the `TilingInterface` contract change
lets the tiler emit a correct, **load-bearing** strided writeback for a real
slice op, on BOTH the `tile_sizes` and `numThreads` offset paths. "Load-bearing"
is proven **ONLY via sub-case A** — a strided dim of size ≥ 2; a size-1 strided
dim is vestigial and is NOT a capability proof. GO does **NOT** prove
`m[0::2,0::2]=True` compiles. That requires (i) dynamic `tensor<?x?xi8>`
(runtime `ceildivi`, unproven), (ii) IREE anchor selection + `computeOp` filter
widening (out of scope, §9), and (iii) end-to-end lowering survival (§8 step 3,
unproven until G4). These are stated as a **separate, unproven boundary**, not
folded into the GO claim.

**Why this gate is meaningful (unlike the refuted plan's):** both halves are
*achievable and checkable*. There is no "fusion never fired" mode that would
make G4 fail for the wrong reason.

**NO-GO modes (each distinct):**
- G3 fails (impl cannot produce the IR) → mechanism/architecture wrong; revisit
  (c) vs (b); do NOT proceed to EXEC.
- G3 holds but ONLY via sub-case B (size-1, vestigial) → **FALSE GREEN**: the
  stride channel is unproven (a hardcoded `[1,1]` passes B). The capability is
  NOT shown; re-derive / re-pick the probe.
- G3 holds, EXEC wrong cells (G4a/G4b) → silent miscompile in the
  offset/size/stride composition (R2's danger); do NOT commit; re-derive §3.
- G3 holds, COMPILE fails (memref→LLVM drops/rejects the stride, §8 step 3) →
  the IR is correct but the lowering stack rejects it; record the next blocking
  layer; the contract change is still proven at IR level.

**On GO:** Phase 0 answers YES — the contract change is sufficient for the tiler
to emit a correct strided writeback for a real slice op. Proceed to the
IREE-side integration (anchor selection + filter widening + dynamic iteration
domain, out of scope here) informed.

---

## 11. Risks (top 2 → review agent; plus the rest)

1. **[RESOLVED in v2 — was HIGHEST UNCERTAINTY] The size convention.** A1's
   `resultSizes = iterSizes*stride` (the "span") produced IR that cannot verify
   (`verifyInsertSliceOp`, `TensorOps.cpp:2885-2896`, is sizes-only). **Fixed to
   `resultSizes = iterSizes`** (the source tile shape; R1 F1, re-derived in §3).
   Under the corrected convention the no-overlap contract
   (`TilingInterface.td:146-147`) is satisfied for all stride ≥ 1, tile ≥ 1
   (point-set gap = `str ≥ 1`, §3) — the only contract-violating variant was the
   discarded `*stride` span. No longer a risk.

2. **[NARROWED in v2 — was HIGH] EXEC lowering survival.** A1 feared the
   `allStridesOne` gate (`:672-674`) might bail; F4 traced the real path (§8):
   `scf.forall` bufferization ignores strides, and `parallel_insert_slice`
   bufferization (`ParallelInsertSliceOpInterface::bufferize` `:969-1027`)
   forwards strides verbatim into `memref.subview` + `memref.copy` — **no gate**.
   So bufferization preserves the stride by construction. The **residual is
   memref→LLVM** (§8 step 3), now **narrowed by R2 §4** (re-verified
   `MemRefToLLVM.cpp:1263-1277`): a strided `memref.copy` is **not rejected** — it
   lowers to a correct, scalar element-wise runtime copy. So the *compile* path is
   not a wall; the residual is **performance** (scalarized copy, cf. LLVM #51660),
   not correctness, and the one COMPILE-fail mode is the pipeline **omitting**
   `-finalize-memref-to-llvm` (m2 — illegal `memref.copy`/`memref.subview`), which
   still proves G3. End-to-end cell correctness remains G4-proven only.

3. **[MEDIUM, DEFERRED — F5] Rank-reduction.** `tensor.insert_slice` is
   rank-reducing; `getDroppedDims()` (`TensorOps.cpp:3217-3219`) calls
   `::getDroppedDims(getSourceType().getShape(), getMixedSizes())` (`:142-180`):
   a size-dim is *dropped* iff it is static-1 and does not correspond to a
   preserved source dim. This creates a **rank-mismatch A1 under-weighted**: the
   iteration domain is the *source* rank, but `getResultTilePosition` must
   return *dest*-rank vectors (the forall `shared_out` is dest-typed). PadOp
   (`TensorTilingInterfaceImpl.cpp:24-84`) never faces this — its iter domain =
   result rank (identity, `:33-44`). The impl rule (deferred, stated here): for
   **preserved** dims `rOff[d]=base[d]+iter_off[d]*str[d]`, `rSize[d]=iter_size[d]`;
   for **dropped** dims `rOff=base[d]`, `rSize=1`. **The G3/G4 probe is
   RESTRICTED to non-rank-reduced** (`sizes` all equal source shape, no static-1
   drops); rank-reduced insert_slice is out of scope (§9) with the rule on record.

4. **[HIGH — F3, on the GO-authorized path] Blast radius.** Giving
   `insert_slice` `TilingInterface` flips ~30 `dyn_cast`/`isa` sites from skip to
   match, including IREE's greedy GPU distributor
   (`GPUGreedilyDistributeToThreads.cpp:139`, the real bug's pipeline) and the
   `TileAndFuseUtils` worklists (`:40, :78, :90, :271, :394`). Containment is
   mandatory and specified in §4a (marker/allow-list gate). Not a G3/G4 blocker
   (the probe is a pure transform-interpreter test) but it lands squarely on the
   integration GO authorizes.
**Resolved (no longer a risk):** the Stage-1 RED is confirmed clean. The
`transform.structured.tile_using_forall` op's `apply`
(`LinalgTransformOps.cpp:3942`) → `tileToForallOpImpl` (`:3973`) →
`dyn_cast<TilingInterface>(target)` at `:3899-3900` (silenceable failure on a
non-`TilingInterface` op) → `scf::tileUsingSCF` at `:3919`. So today the RED is
a clean "not TilingInterface" rejection (fixed by Stage 3), not a copy-anchor
rejection.
- The producer-fusion stretch (5.3) routes through `:1006-1007` (unit) and may
  also touch `:1565-1566` if the fill's init is non-trivial; if it routes
  through `:1502-1504` with a strided candidate, that guard must flip (pulls the
  producer dest-extract path into scope). Stage 5.3 must trace the exact sites.
- Upstream-vs-local-fork is undecided; this plan is implementable as a local
  vendored patch regardless, but upstreaming the `TilingInterface` contract
  change is a multi-week review battle (feasibility doc §1).

## 12. Reasoning & Justification (the WHY of every v2 change)

> Per-finding OLD/NEW/evidence changelog: `approach1_contract_rework_a2_changes.md`.
> This section is the *decision rationale* behind each correction.

**F1 — why `resultSizes = iterSizes`, not `iterSizes*stride`.** `insert_slice`
is the *inverse* of `extract_slice`. For `extract_slice`, `sizes` is the span
read out; for `insert_slice`/`parallel_insert_slice`, `sizes` is the number of
*source* elements placed (element $j$ → dest $\text{offset}+j\cdot\text{stride}$).
The verifier (`verifyInsertSliceOp`, `TensorOps.cpp:2885-2896`) computes the
expected source type as `ExtractSliceOp::inferResultType(dstType, staticSizes)` —
**sizes only, no strides** — so the source/`%tile` shape must equal `sizes`. The
dest span is implicit (`offset+(size-1)*stride`, checked in-bounds at `:2910`).
A1 applied *extract*-slice reasoning (`sizes`=dest span) to an *insert*-slice op.
The one-line fix is `resultSizes[d] = iterSizes[d]`; it also makes the
no-overlap contract hold (point-set gap `= str ≥ 1`).

**F2 — why the anchor IR is `[0,0][2,4][2,1]`.** Same authority: `sizes` must
equal the `<2x4>` source shape. A1's `[4,4]` failed both the type check
(`inferResultType(dest<4x4>,[4,4])`≠`<2x4>`) and in-bounds (`(4-1)*2+1=7>4`).

**A/B reframe — why the genuine scatter (sub-case A) is the only capability
proof.** Placement is $\text{offset}+j\cdot\text{stride}$; on a size-1 strided
dim, $j\in\{0\}$ so the stride is inert (vestigial) and a hardcoded `[1,1]`
writeback passes identically. The stride channel is *load-bearing* only when a
strided dim has size ≥ 2 (tile row 1 lands at a stride-dependent dest index that
unit-stride gets wrong). A1's tile-size-1 primary was therefore a false-green
for the capability claim; the full-source scatter (size-2 on the strided dim) is
the real probe. (Verified against the verifier — it checks sizes/in-bounds, not
stride meaningfulness.)

**F3 — why gate the interface's *use*, not its *registration*.** Registration is
global, so every `dyn_cast<TilingInterface>` site newly matches `insert_slice`.
The marker gate returns `failure()` from `getIterationDomain` on unmarked ops:
the cast succeeds, the immediate next call fails, the consumer no-ops — exactly
today's behavior, with only the explicit transform probe (which sets the marker)
exercising the new path. Omitting the operand-tile/fusion methods makes fusion
consumers safe by default. This is the firewall that lets a global interface
change ship behind a flag without regressing the real bug's pipeline.

**F4 — why the stride survives bufferization and the real wall is memref→LLVM.**
The G3 writeback is a `parallel_insert_slice` *inside* `scf.forall`, on a
*different* bufferization model than the standalone `InsertSliceOp` A1 cited.
`ForallOpInterface::bufferize` ignores strides; `ParallelInsertSliceOpInterface::bufferize`
forwards `getMixedStrides()` verbatim into `memref.subview`+`memref.copy`. No
gate. So the G4 risk is not "bufferization bails" but "memref→LLVM drops/rejects
the strided subview" — a narrower, honest statement of the unverified half.

**F5 — why rank-reduction is deferred with the rule stated.** `getDroppedDims`
(`TensorOps.cpp:3217-3219`) makes the iteration domain (source rank) and result
position (dest rank) *different ranks* — a mismatch PadOp (the only precedent)
never faces (its iter domain = result rank). Inventing both strided AND
rank-mismatched result-tile semantics in one probe is unjustified; the rule is
stated (preserved dims compose; dropped dims get size 1 / offset = base) and the
probe restricted to non-rank-reduced.

**6a — why numThreads is promoted to core G4b.** The `numThreads` branch computes
per-tile offsets via *different code* than `tile_sizes` (`:473-544`, with
residual/boundary `min/max`), so a `tile_sizes`-only green does not cover the
real-IREE offset path. Promotion is cheap (one lit+EXEC on the same harness) and
converts the `iv*T*S` "by construction" math from inference to evidence — but
the case must be a genuine scatter (per-tile size ≥ 2) to avoid being a
vestigial-stride false green itself.

**6b — why GO claims only IR-level + static-EXEC capability.** The real dispatch
is dynamic `tensor<?x?xi8>` wired through IREE anchor selection + `computeOp`
filters. None of that is exercised by a static transform-interpreter probe.
Folding it into GO would be a false claim; stating dynamic + IREE-wiring as a
separate, unproven boundary is the honest scope line.

---

## Appendix — citation index (all personally opened this session)

- `include/mlir/Interfaces/TilingInterface.td:118-162` (`getResultTilePosition`,
  no strides), `:202-301` (operand-tile methods, out of scope).
- `lib/Dialect/Linalg/Transforms/TilingInterfaceImpl.cpp:156-208`
  (`getMappedOffsetAndSize`, `AffineDimExpr` enforcement `:169-171`),
  `:212-231` (`getIterationDomainTileFromOperandTiles`, no
  `isProjectedPermutation` — review FLAW 4 confirmed), `:235-259` (linalg
  `getResultTilePosition`, passthrough for fill), `:261-288`
  (`getIterationDomainTileFromResultTile`, `isProjectedPermutation` `:274`),
  `:1029`/`:1499` (pack/unpack `getResultTilePosition`).
- `lib/Dialect/Tensor/IR/TensorTilingInterfaceImpl.cpp:24-84` (PadOp model — the
  impl template), `:311-316` (registration — where insert_slice attaches).
- `lib/Dialect/SCF/Transforms/TileUsingInterface.cpp:336-364` (the two typedefs),
  `:388-465` (`generateLoopNestUsingForOp`, `:447-448`), `:473-544`
  (`getTileOffsetAndSizesWithForAllOp`, numThreads offset `:491`), `:556-624`
  (`generateLoopNestUsingForallOp`, **`:616-617`**), `:687-724` (`generateLoopNest`),
  `:726-788` (`createInitialTensorsForTiling`, `:734` getOrCreateDestinations),
  `:848-856` (static `getResultTilePosition` helper), `:1112-1249` (`tileUsingSCF`,
  body lambda `:1158-1226`, `:1211-1214` getResultTilePosition call,
  `:1241` generateLoopNest), `:951-955`/`:1006-1007` (fusion writeback),
  `:1502-1504` (producer rejection), `:1565-1566` (producer dest-extract),
  `:2311`/`:2313-2317` (consumer candidate read/reject), `:2366-2367` (consumer
  dest-extract).
- `lib/Dialect/Tensor/Transforms/SwapExtractSliceWithProducerPatterns.cpp:25-60`
  (`replaceExtractSliceWithTiledProducer`, guard `:31-33`),
  `:62-115` (`replaceInsertSlicesWithTiledConsumer`, guard `:99-101`). **Path
  corrected: `Tensor/Transforms/`, not `Linalg/Transforms/`.**
- EXEC templates: `test/Integration/Dialect/Vector/CPU/transfer-write.mlir:1-4`
  (`-test-lower-to-llvm | mlir-runner -shared-libs=%mlir_c_runner_utils`),
  `test/Integration/Dialect/Vector/CPU/ArmSVE/contraction.mlir:1-5`
  (transform-interpreter compile/run split).

### A2 additional citations (opened this rework pass — F3/F4/F5)
- **Verifier (F1/F2 authority):** `lib/Dialect/Tensor/IR/TensorOps.cpp:2885-2896`
  (`verifyInsertSliceOp`, sizes-only via `inferResultType`), `:2898-2917`
  (`InsertSliceOp::verify` + in-bounds `:2910-2912`), `:3960-3982`
  (`ParallelInsertSliceOp::verify`).
- **Rank-reduction (F5):** `TensorOps.cpp:142-180` (`::getDroppedDims` rule),
  `:3217-3219` (`InsertSliceOp::getDroppedDims`), `:3991-3993`
  (`ParallelInsertSliceOp::getDroppedDims`).
- **F4 real lowering path:** `lib/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.cpp:1209-1296`
  (`ForallOpInterface::bufferize` — ignores strides; `:1209-1212` "terminators
  analysis only"), `lib/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.cpp:949-1037`
  (`ParallelInsertSliceOpInterface::bufferize` — forwards strides to
  `memref.subview` `:998-1002` + `memref.copy` `:1005-1006`, registered
  `:1209-1210`), `:655-733` (`InsertSliceOpInterface` standalone — the
  `allStridesOne` analysis gate `:672-674` is HERE, off the writeback path;
  `:694-698` hostility comment).
- **F3 blast-radius sites:** `compiler/src/iree/compiler/Codegen/Common/GPU/GPUGreedilyDistributeToThreads.cpp:114-154`
  (`processRegion` walk, `dyn_cast<TilingInterface>` at `:139`, `tileToThreads`
  at `:145`), `compiler/src/iree/compiler/Codegen/Common/TileAndFuseUtils.cpp:40,78,90,271,394`
  (fusion worklist casts).
- **numThreads offset branch:** `lib/Dialect/SCF/Transforms/TileUsingInterface.cpp:473-544`
  (`getTileOffsetAndSizesWithForAllOp`, `offsetExpr=d0+d1*s0` `:491`,
  residual `min/max` `:514-538`), `:556-624`
  (`generateLoopNestUsingForallOp`; `innerDestinationTensors=getRegionOutArgs()`
  region iter args at `:597`; writeback zip `:613-621`).
- **PadOp precedent (iter=rank identity):** `lib/Dialect/Tensor/IR/TensorTilingInterfaceImpl.cpp:24-84`
  (`getIterationDomain` `:33-44`, identity `getResultTilePosition` `:57-66`,
  `generateResultTileValue` `:79-83`), registration `:311-316`.
