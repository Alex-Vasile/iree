# Approach 1 (CONTRACT) Plan — Final Adversarial Review (R2)

> Reviewer: senior MLIR/IREE compiler architect (R2 — final adversarial pass).
> Target: `approach1_contract_phase1_plan.md` **v2 (A2 rework)** + A2's changelog
> `approach1_contract_rework_a2_changes.md`.
> Method: every `file:line` below was opened personally in the
> `~/Developer/iree` / `third_party/llvm-project` tree this session. `[INFERENCE]`
> marks reasoning-only claims. No code changed, no builds run. A2's fixes were
> re-verified against source — the changelog was not trusted.

---

## 1. Verdict — CONVERGED (ready to execute)

The plan has converged on every blocking correctness issue. **All six v2 fixes
are confirmed correct against the source** (§2). The hardest open question —
A2's `[INFERENCE]` on whether memref→LLVM rejects a strided `memref.copy`
destination — is **resolved positively**: the stride survives end-to-end (§4).
The numThreads G4b promotion is sound and its math is correct (§2.3). The
GO/NO-GO gate is well-defined and free of the prior "fusion-never-fired"
ambiguity.

This is a GO to execute. The four remaining items in §3 are **text-precision /
harness corrections** that an implementer will hit and resolve during execution;
none requires a rework round and none blocks the mechanism. They are listed as
**execution risks**, not A3 prerequisites, in §5.

**Why CONVERGED rather than NEEDS-A3:** the two issues that come closest to
meriting a round (§4a's marker-gate mis-description, §8's missing
`-convert-memref-to-llvm`) are both (a) localized to *descriptive* text whose
*correct* form is already partially present or obvious from the cited code, and
(b) self-correcting at execution time (an implementer reading `getIterationDomain`'s
return type will not gate there; the EXEC compile will reject the bare pipeline).
A rework round's only yield would be cosmetic text hygiene. The substantive
engineering judgment — mechanism, size math, scatter reframe, lowering trace —
is settled and verified.

---

## 2. Fix verification (every v2 fix confirmed or refuted)

### 2.1 F1 — `resultSizes = iterSizes` everywhere — CONFIRMED (re-derived)

Authority re-opened: `verifyInsertSliceOp` (`lib/Dialect/Tensor/IR/TensorOps.cpp:2885-2896`):

```cpp
RankedTensorType expected =
    ExtractSliceOp::inferResultType(dstType, staticSizes);  // :2891-2892
if (expectedType) *expectedType = expected;
return isRankReducedType(expected, srcType);                // :2895
```

`sizes` is the only geometry fed to the type check — **strides are absent**.
A2's `resultSizes = iterSizes` (source tile shape) is therefore forced.

**Hand re-derive of the sub-case A per-tile writeback** (the load-bearing probe),
emitted at `generateLoopNestUsingForallOp:616-621`:

```mlir
tensor.parallel_insert_slice %tile into %o0[0, 0] [2, 4] [2, 1]
  : tensor<2x4xi32> into tensor<4x4xi32>
```

- Type check: `inferResultType(tensor<4x4>, [2,4])` = `tensor<2x4>` == `%tile`
  type ✓ (`:2891-2895`).
- In-bounds: `verifyInBoundsSlice` (`:2910-2912`, applied to dest type) — last
  accessed dest index per dim = `offset + (size-1)*stride`. Dim-0:
  `0 + (2-1)*2 = 2 < 4` ✓; dim-1: `0 + (4-1)*1 = 3 < 4` ✓. (Stride-element
  traversal confirmed by the `for (i...) offset += stride` fold at `:2632`/`:2638`.)

**The writeback is valid IR.** Confirmed in the call-chain box (`plan §1.3`,
`resultSizes = iterSizes`), the §3 math, and Stage 3.1. The v1 `iterSizes*stride`
inversion is gone everywhere I checked. A1's own reasoning companion
(`approach1_contract_rework_a1_reasoning.md:248`) still shows the OLD
`sizes[d]*str[d] = [2,4]` — but that is a *companion* doc, not the plan; the
plan itself is clean.

### 2.2 F2 — Case-1 anchor `sizes [2,4]` — CONFIRMED

`insert_slice %filled<2x4> into %dest<4x4>[0,0][2,4][2,1]`:
`inferResultType(<4x4>,[2,4])` = `<2x4>` == source (`:2891`) ✓; in-bounds ✓
(same as §2.1). v1's `[4,4]` is gone. Sub-case A's `tile_sizes [2,4]` and B's
`tile_sizes [1,4]` both derive from a verified anchor.

### 2.3 numThreads G4b promotion — CONFIRMED SOUND (and the iv*T*S math is correct)

This was the highest-stakes v2 change. Three independent checks:

**(a) The ceilDiv is real and computed by the caller.** A2's claim that
`givenTileSize = ceilDiv(range, numThreads)` is baked in *before*
`getResultTilePosition` runs is verified at
`getUserTileSizesAndNumThreads` (`TileUsingInterface.cpp:120-138`):

```cpp
// tileSize = ceilDiv(niters, numThreads)
AffineExpr numItersExpr = (s1 - s0);            // :127  (range.size - range.offset)
AffineExpr tileSizeExpr = numItersExpr.ceilDiv(s2); // :128
...
tileSizes[index] = makeComposedFoldedAffineApply(..., tileSizeExpr,
                                                 {range.offset, range.size, nt}); // :135-136
```

`tileUsingSCF:1126` calls this, then `:1241` passes `givenTileSizes` into
`generateLoopNest` → `generateLoopNestUsingForallOp:602`. So the effective
size IS `ceilDiv(range, numThreads)` by the time
`getTileOffsetAndSizesWithForAllOp` runs. The plan's §7 "by construction"
claim is grounded, not asserted.

**(b) The offset formula is a distinct code path.** `getTileOffsetAndSizesWithForAllOp`
(`:473-544`) only enters the numThreads branch when `numThreads` is non-empty
(`:479-482`); otherwise it falls to `getTileOffsetAndSizes` (the `tile_sizes`
path). The numThreads branch uses `offsetExpr = d0 + d1*s0` (`:491`,
`offset = loopRange.offset + iv*givenTileSize`) **plus** residual/boundary
`min/max` handling (`:514-538`) that the `tile_sizes` path does not exercise. A
`tile_sizes`-only green therefore does NOT cover this path. Promotion is
justified.

**(c) The math is correct and G4b is a genuine scatter, not vestigial.**
Re-derive the G4b case (`source<4x4> → dest<8x4>`, stride `[2,1]`,
`num_threads [2]`): dim-0 `range=4, nt=2 → tileSize=ceilDiv(4,2)=2`; 2 tiles,
each size `[2,4]`. Per-tile iteration offset `iv*2` (`:491`). The insert_slice
`getResultTilePosition` then composes `resultOffset[0] = base[0] + offset*stride[0]
= 0 + (iv*2)*2 = iv*4`. Tile 0 → dest rows `{0,2}`; tile 1 → `{4,6}`. **Each
per-tile writeback is size-2 on the strided dim** → within-tile row 1 lands at
`0 + 1*2 = 2` (tile 0) / `4 + 1*2 = 6` (tile 1). The stride is load-bearing
*inside* each tile, so G4b is **not** a vestigial-stride false-green (the trap
A2 identified in the A/B reframe). The promotion is sound.

### 2.4 A/B scatter reframe — CONFIRMED

- Sub-case A (`tile_sizes [2,4]`, 1-tile genuine scatter) is the PRIMARY G3/G4
  assertion: tile row 1 → dest row `0 + 1*2 = 2`; a hardcoded `[1,1]` writeback
  would place it at dest row **1**. Stride load-bearing ✓.
- Sub-case B (`tile_sizes [1,4]`, size-1 strided dim) is demoted with the
  false-green note: `j∈{0}` → placement = `offset+0 = offset`, identical to
  unit stride. A hardcoded `[1,1]` passes B byte-for-byte. ✓
- §10 GO claims capability **only via A** ("Load-bearing is proven ONLY via
  sub-case A"), with an explicit NO-GO mode "G3 holds but only via B → FALSE
  GREEN" (`plan §10`). ✓

**Applying the scatter-vs-offset lens to G4b:** as derived in §2.3(c), G4b is a
genuine within-tile scatter (size-2 on the strided dim in *every* tile), not a
vestigial one. G4b is not a second false-green — it is the stronger probe.

### 2.5 F4 — EXEC lowering trace — CONFIRMED, and the `[INFERENCE]` is RESOLVED

This is A2's least-sure item and the assignment asked me to press hardest here.
Three hops, all opened:

**Hop 1 — `scf.forall` bufferization ignores strides.** `ForallOpInterface::bufferize`
(`lib/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.cpp:1243-1296`):
replaces each `shared_out` region arg with `to_tensor(memref)` (`:1265-1267`)
and `mergeBlocks` (`:1289-1290`). It never inspects the terminator's strides
(the `parallel_insert_slice` is carried verbatim into the new block). ✓
(Confirmed: the `:1209-1212` "terminators analysis only" comment is on the
terminator *models*, not the forall bufferize itself.)

**Hop 2 — `parallel_insert_slice` bufferization forwards strides verbatim.**
`ParallelInsertSliceOpInterface::bufferize`
(`lib/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.cpp:969-1027`):

```cpp
MemRefType subviewMemRefType =
    memref::SubViewOp::inferRankReducedResultType(
        parallelInsertSliceOp.getSourceType().getShape(), destBufferType,
        parallelInsertSliceOp.getMixedOffsets(),
        parallelInsertSliceOp.getMixedSizes(),
        parallelInsertSliceOp.getMixedStrides());        // :992-997
Value subview = memref::SubViewOp::create(rewriter, loc, subviewMemRefType,
    *destBuffer, ...getMixedOffsets(), ...getMixedSizes(),
    ...getMixedStrides());                                 // :998-1002
if (failed(options.createMemCpy(rewriter, loc, *srcBuffer, subview))) // :1005-1006
```

**No stride gate.** The `[2,1]` flows into a strided `memref.subview` and a
`memref.copy` whose **target** is non-contiguous. A2's correction (the
`:672-674` `allStridesOne` gate is on the *standalone* `InsertSliceOp` model at
`:655-733`, off this path) is right.

**Hop 3 — memref→LLVM (the `[INFERENCE]`, now resolved).** The strided
`memref.copy` lowers in `MemRefCopyOpLowering::matchAndRewrite`
(`lib/Conversion/MemRefToLLVM/MemRefToLLVM.cpp:1257-1278`):

```cpp
auto isContiguousMemrefType = [&](BaseMemRefType type) { ... };  // :1263-1272
if (isContiguousMemrefType(srcType) && isContiguousMemrefType(targetType))
  return lowerToMemCopyIntrinsic(op, ...);   // flat LLVM::MemcpyOp :1274-1275
return lowerToMemCopyFunctionCall(op, ...);  // generic MemrefCopyFn :1277
```

The `memref.copy` op itself only requires `SameOperandsElementType,
SameOperandsShape` (`MemRefOps.td:578-579`) — no contiguity trait — so it
*verifies* with a strided target (same shape `<2x4>`, strided layout). At
lowering, a strided target fails the `isContiguousMemrefType` test and takes the
`lowerToMemCopyFunctionCall` branch (`:1191-1255`), which calls the generic
runtime `LLVM::lookupOrCreateMemRefCopyFn` (`:1241`, created in-module) — an
**element-wise copy that respects the target's strides** (the file header comment
at `:1138-1139` states it explicitly: "For non-identity layouts, the copy is
lowered to a call to the generic `MemrefCopyFn`"). The strided `memref.subview`
itself is bog-standard and lowers through `TypeConverter`'s strided-form path
(`TypeConverter.cpp:490` only fails on *non-strided* maps, which a subview is
not).

**Conclusion: a strided `memref.copy` destination is NOT rejected. It lowers to
a correct (if non-vectorized, slow) element-wise runtime copy that honors the
strides. G4 is reachable.** A2's `[INFERENCE]` is resolved — positively. The
plan's §8 statement that "bufferization preserves the stride by construction
… the risk is entirely in memref→LLVM" is now grounded: the risk is a
*performance* one (scalar strided copy, cf. §9 LLVM #51660), not a *correctness
or compile* one. The headline fear — "a hard rejection downstream of
bufferization makes G4 unreachable" — does **not** materialize.

### 2.6 F5 — rank-reduction restriction — CONFIRMED

`getDroppedDims` (`TensorOps.cpp:142-180`): a size-dim is dropped iff static-1
*and* the reduced-shape dim is not also 1 (`:149-176`). The plan states the
preserved/dropped rule (`§11 #3`) and restricts the probe to non-rank-reduced.
Sub-case A is verified non-rank-reduced: source `<2x4>`, sizes `[2,4]` — both
non-unit, so `shapePos` walks to `-1` with no `droppedDims.set` (matching
`InsertSliceOp::getDroppedDims` at `:3217-3219`). The probe avoids rank
reduction cleanly. ✓

### 2.7 Blast-radius containment (§4a) — PARTIALLY confirmed (one real flaw, see §3.1)

Both cited danger sites are real and re-verified:
- `GPUGreedilyDistributeToThreads.cpp:139` — `dyn_cast<TilingInterface>(op)`
  inside `processRegion` walk (`:114-154`), routed to `tileToThreads` (`:145`).
  Confirmed in the real bug's GPU distribution path.
- `TileAndFuseUtils.cpp` — `:40` (`getDefiningOp<TilingInterface>` producer),
  `:78`/`:90` (`isa<TilingInterface>` producer/user in `collectTiledAndFusedOps`),
  confirmed.

The producer-fusion containment (firewall: omit `generateResultTileValue`) is
sound: `fuseProducersOfSlices` calls `scf::tileAndFuseProducerOfSlice`
(`TileAndFuseUtils.cpp:57`), which needs that method; its absence → `failure()`
→ `continue` (`:58-60`). The consumer containment via `tileToThreads` is
best-effort and does bail on failure (`GPUGreedilyDistributeToThreads.cpp:96-98`,
40-42 comment "If tiling fails this returns silently"). **But** the *mechanism
§4a describes* has a technical flaw — see §3.1.

---

## 3. Remaining holes (R2-specific findings, ranked by severity)

### 3.1 [MEDIUM] §4a marker-gate names an impossible gate point

§4a (and A2's F3 entry) states the impl's "`getIterationDomain` /
`getTiledImplementation` / `getResultTilePosition` early-return `failure()`
unless a marker is present." **`getIterationDomain` cannot return `failure()`.**
Its signature (`include/mlir/Interfaces/TilingInterface.td:80-85`) is:

```tablegen
/*retTy=*/"::mlir::SmallVector<::mlir::Range>",
/*methodName=*/"getIterationDomain",
/*args=*/(ins "::mlir::OpBuilder &":$b),
/*defaultImplementation=*/"return {};"
```

It returns a `SmallVector<Range>` — no `LogicalResult`. (PadOp's model at
`TensorTilingInterfaceImpl.cpp:33-44` confirms the non-failing shape.) A literal
"gate getIterationDomain on the marker" is ill-formed.

**Containment is still achievable** — the gate must instead live on
`getTiledImplementation` (returns `FailureOr<TilingResult>`, `:107`) and/or
`getResultTilePosition` (returns `LogicalResult`, `:149`). I traced it:
`tileToThreads` (`:94`) → `tileConsumerAndFuseProducersUsingSCF` →
`tileUsingSCF` calls `getIterationDomain` (`:1122`, gets valid source-shape
ranges), sets up the forall, then the body lambda calls `getTiledImplementation`
(`:1193-1196`); an unmarked impl returning `failure()` there → the lambda fails
(`:1197-1199`) → `generateLoopNestUsingForallOp:609-610` `notifyMatchFailure` →
`tileUsingSCF:1244` failure → `tileToThreads:96-98` silent return. So the
*effect* §4a wants holds; only the *named site* is wrong.

This is a descriptive error an implementer will catch on first compile (the
method has no failure to return). It does not block execution, but the spec
should be corrected so the firewall is unambiguous: **gate at
`getTiledImplementation`/`getResultTilePosition`, not `getIterationDomain`.**

### 3.2 [LOW-MEDIUM] §8 explicit EXEC pipeline omits `-convert-memref-to-llvm`

The pipeline block (`plan §8, lines 687-696`) lists `-convert-scf-to-cf`,
`-convert-cf-to-llvm`, `-convert-arith-to-llvm`, `-convert-func-to-llvm`,
`-convert-index-to-llvm`, `-reconcile-unrealized-casts` — but **not**
`-convert-memref-to-llvm`. The `MemRefCopyOpLowering` that §4 proved is the
strided-copy handler is registered by `-convert-memref-to-llvm`; without it,
`memref.copy`/`memref.subview` survive into LLVM-dialect conversion and are
rejected (unknown op) — a *compile* failure unrelated to strides.

The cited template (`test/Integration/Dialect/Vector/CPU/transfer-write.mlir:1`)
uses the monolithic `-test-lower-to-llvm`, which pulls in all `-convert-*`
passes including memref. The plan does flag `-test-lower-to-llvm` as the
fallback (`plan §8 line 746`). So this is a harness-precision gap with an
existing escape hatch, not a blocker. **Execution note:** if the explicit
pipeline is used as-written, G4 will fail at the LLVM conversion with an
"illegal op" error on `memref.copy`, *not* a stride error — diagnose by adding
`-convert-memref-to-llvm` (or switching to `-test-lower-to-llvm`).

### 3.3 [LOW] G4a 1-iteration forall is foldable → G3 FileCheck fragility

Sub-case A tiles a `<2x4>` source with `tile_sizes [2,4]` → exactly **1
iteration**. A 1-iteration `scf.forall` is a canonicalization target; if it
folds, the writeback may surface as a plain `tensor.insert_slice` (not
`parallel_insert_slice`) and the §3 FileCheck (which pins
`parallel_insert_slice %tile into %o0[0,0][2,4][2,1]`) would not match — a
FileCheck RED that is *not* a mechanism failure. The plan acknowledges this
(`§3 line 310-313`) and offers the 2-tile genuine-scatter case as the
substitute. Note that substitute *is* the G4b shape (`<4x4>→<8x4>`), so G4a and
G4b converge if folding bites. Minor robustness risk for the G3 (IR-level)
gate; G4 (EXEC) is unaffected.

### 3.4 [LOW] §4a firewall is best-effort, not hard

The `tileToThreads` containment depends on `GPUGreedilyDistributeToThreads`
being best-effort ("If tiling fails this returns silently",
`GPUGreedilyDistributeToThreads.cpp:40-42,96-98`). That is true *today*; the
hard firewall is firewall 2 (the IREE `isComputeOp`/anchor allow-list, `§4a`
item 2, correctly scoped as the integration PR). If a future change made
`tileToThreads` *not* bail (or *not* route through `getTiledImplementation`),
an unmarked insert_slice would be mis-tiled. The plan correctly identifies the
allow-list as defense-in-depth; flag it so the integration PR (out of this
plan's scope, `§9`) lands the allow-list before the marker is ever dropped.

---

## 4. The memref→LLVM lowering question — resolved

**Resolved positively (no `[INFERENCE]` left).** A strided `memref.copy`
destination (the exact artifact `ParallelInsertSliceOpInterface::bufferize`
emits for a `[2,1]` writeback) lowers correctly:

- The op verifies: `memref.copy` needs only `SameOperandsElementType,
  SameOperandsShape` (`MemRefOps.td:578-579`); a strided-layout target of the
  same shape is legal.
- The lowering forks (`MemRefToLLVM.cpp:1274-1277`): contiguous×contiguous →
  flat `LLVM::MemcpyOp`; **anything strided → `lowerToMemCopyFunctionCall`** →
  generic `LLVM::lookupOrCreateMemRefCopyFn` (`:1241`, created in-module), an
  element-wise copy honoring both operands' layouts.
- The strided `memref.subview` is ordinary; `TypeConverter` only rejects
  *non-strided* maps (`TypeConverter.cpp:490-492`), which a subview never is.

**Net:** there is no hard rejection downstream of bufferization for the static
EXEC case. G4 is reachable and should produce correct cells (slowly — the copy
is scalarized, consistent with the plan's §9 LLVM #51660 note). The plan's §8
claim that "the risk is entirely in memref→LLVM" is, on inspection, a
*performance* caveat rather than a *correctness/compile* one for the static
probe. **This was the single highest-uncertainty item in the whole loop; it is
now grounded, and it resolves in the plan's favor.**

The one residual: the explicit pipeline as written omits the pass that owns
this lowering (§3.2). That is a harness bug, not a lowering-wall.

---

## 5. Top execution risks (CONVERGED → no A3 round)

Since the verdict is CONVERGED, the numbered A3 changes are replaced by the
risks an implementer must watch. In priority order:

1. **Apply the §4a gate-point correction first** (§3.1). Gate the marker on
   `getTiledImplementation`/`getResultTilePosition` (`TilingInterface.td:107,149`),
   *not* `getIterationDomain` (`:80-85`). `getIterationDomain` should return the
   true source-shape domain unconditionally; the marker only suppresses tiling
   execution. Verify the containment by running `GPUGreedilyDistributeToThreads`
   on a dispatch containing an *unmarked* insert_slice and confirming it is
   left untouched (Stage 0 smoke).

2. **Fix the §8 EXEC pipeline** (§3.2): add `-convert-memref-to-llvm` to the
   explicit pass list, or run `-test-lower-to-llvm` (monolithic, the
   `transfer-write.mlir` template). If G4 fails with an "illegal op: memref.copy"
   rather than wrong cells, this is the cause — it is *not* a stride/contract
   failure.

3. **G4a FileCheck robustness** (§3.3): write the G3 FileCheck to match the
   `[2,4][2,1]` geometry under *either* `parallel_insert_slice` or
   `insert_slice`, or drive sub-case A with `-canonicalize` disabled for the IR
   check; keep the 2-tile `<4x4>→<8x4>` case ready as the non-foldable
   alternative (it doubles as G4b).

4. **The "best-effort" containment is load-bearing** (§3.4): the marker-gate
   must make `getTiledImplementation` fail *fast* (no partial forall leak).
   Confirm by inspection that `generateLoopNestUsingForallOp`'s
   `notifyMatchFailure` path (`:609-610`) rolls back the forall it created at
   `:585` when `tiledBodyFn` fails — a leaked partial forall in the real GPU
   pipeline would be a silent regression. The IREE allow-list (firewall 2) is
   the hard version and must ship in the integration PR before any marker is
   dropped.

5. **DPS sanity at Stage 3** (not a hole, a checkpoint): insert_slice is DPS
   with `getDpsInits = dest` (`TensorOps.td:843,973`), so
   `getOrCreateDestinations` (`TileUsingInterface.cpp:734`) returns `[dest]` and
   the forall `shared_out` is the dest — the per-tile writeback target `%o0`
   (`getRegionOutArgs()`, `:597`) aliases it. This is correct by construction
   and verified, but confirm at Stage 3.3 that the tiled value is the
   `extract_slice %source` (the *source* tile), not a slice of dest — the
   writeback composes source-tile + strided insert into the dest region arg.

6. **Keep the dynamic-size boundary honest** (§9/§10): GO proves IR-level +
   static-EXEC capability only. The real `m[0::2,0::2]=True` dispatch is dynamic
   `tensor<?x?xi8>` (runtime `ceildivi`, `getUserTileSizesAndNumThreads:135`
   produces an affine expr, not a constant) wired through IREE anchor selection
   — unproven by this probe. Do not let a green G4 be reported as "the dispatch
   compiles."

---

## 6. Notes on items the assignment flagged for hunting

- **Other false-greens beyond the scatter one:** none found. The DPS/getOrCreateDestinations
  path is sound (§2, risk 5). The no-overlap contract (`TilingInterface.td:146-147`)
  holds under `resultSizes = iterSizes` (point-set gap `= str ≥ 1`). The
  numThreads case is a genuine scatter (§2.3c), not vestigial.
- **Does TilingInterface change `getOrCreateDestinations`/DPS handling?** No.
  insert_slice is already DPS (`TensorOps.td:843`) with `getDpsInits = dest`
  (`:973`); the interface addition does not alter that. `createInitialTensorsForTiling`
  (`:726-788`) returns `[dest]` via `getOrCreateDestinations` (`:734`); the
  forall `shared_out` and region-iter-arg writeback target are both dest. ✓
- **Is propagate-first/flip-last literally true in the Stage ordering?** Yes.
  Stage 2 threads the channel with unit defaults (zero behavior change); Stage
  3 adds the impl; Stage 5.3 flips G1/G2 last. No rejection guard is removed
  before the channel carries the stride. (The plan's invariant is confusingly
  also labeled "R1" — collides with the review name, but not a logic bug.)
- **Is the GO/NO-GO gate free of "fusion-never-fired" ambiguity?** Yes. The
  mechanism is *initial tiling* of the anchor, not consumer fusion; the Stage-1
  RED is a clean `dyn_cast<TilingInterface>` failure
  (`LinalgTransformOps.cpp:3899-3900`), confirmed by R1 and A1. G4 cannot fail
  for a "fusion didn't fire" reason — there is no fusion to fire.
