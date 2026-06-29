# Reasoning Companion — Approach 1 Contract-Based Rework (A1)

> **Purpose:** durable record of the *chain of thought* behind
> `approach1_contract_phase1_plan.md` — the WHY, not just the conclusions.
> This is the reasoning the plan was built on, so a future reader (or the review
> agent) can audit each decision against the code rather than take the plan's
> word for it.
>
> **Grounding rule (same as the plan):** every claim cites a `file:line` I
> personally opened in the `~/Developer/iree` / `third_party/llvm-project` tree
> this session. Reasoning-only (not directly observed) claims are marked
> `[INFERENCE]`. **No code was changed; no builds were run.** This file does NOT
> modify the plan — it is a companion.

---

## 0. The shape of the problem, restated so the decision is forced

The IREE dispatch `m[0::2,0::2]=True` fails because one-shot bufferization emits
a bare `#hal.descriptor_type<storage_buffer>` copy at dispatch scope, which
`VerifyWorkgroupDistributionPass` rejects (feasibility doc §3.3). The copy is
forced because the write-back is **transposed**: the fill region is strided on
dim-1 (`[1,2]`) while the output store is strided on dim-0 (`[2,1]`) — they
agree on no strided axis, so bufferization cannot alias in place
(`approach1_tensor_level_fusion.md` §3.3, §2.4.1).

The expert review (`expert_review_phase1_plan.md` §1, §3) refuted the SCF-only
consumer-fusion plan and nailed the real wall: **strides can only be
*introduced* at initial tiling** (`generateLoopNestUsingForallOp:616-617` /
`generateLoopNestUsingForOp:447-448`), because consumer fusion reads the
loop-*internal* unit-stride candidate (`getProducingParallelInsertSlice`,
`TileUsingInterface.cpp:2487`) and can therefore only *preserve* a stride that
already exists — it **cannot create** one (review §3 FLAW 3). That collapses the
whole problem to one question:

> **At initial tiling, what is the STRIDE SOURCE that reaches `:616-617`?**

Everything below is an answer to that question, derived by opening the code
rather than reasoning about it in the abstract.

---

## 1. WHY mechanism (c) — insert_slice as a `TilingInterface` anchor

### 1.1 First hypothesis (rejected): the fill is the anchor, so its `getResultTilePosition` carries the stride

IREE's anchor selection picks `linalg.fill` as the last computeOp with a workgroup
tiling level (`TileDispatchUsingForall.cpp:67-76`; `isComputeOp` =
`TilingInterface | UKernelOpInterface`, `Utils.cpp:980-982`). My first instinct
was: tile the fill, add a `resultStrides` out-param to `getResultTilePosition`,
let the fill populate it.

I opened the linalg implementor to check whether a fill *could* return a strided
result position. It cannot:

- `linalg::getResultTilePosition` (`TilingInterfaceImpl.cpp:235-259`) builds
  `resultOffsets`/`resultSizes` from `computeSliceParameters` with the indexing
  map (`:252-255`). For `linalg.fill` the output indexing map is the identity, so
  this is a pure passthrough of the iteration-space `offsets`/`sizes` — there is
  no stride in the computation, and nothing to populate.
- More fundamentally, linalg indexing maps are **projected permutations**. The
  `isProjectedPermutation()` check lives in `getIterationDomainTileFromResultTile`
  (`:274`), and `getMappedOffsetAndSize` (`:156-208`) enforces the same property
  by requiring each map result be an `AffineDimExpr` (`:169-171`, `return
  failure()` otherwise). A projected permutation maps iteration dims to result
  dims by *identity-with-reordering*; it has no coefficient slot for a stride.
  [Note: the prior docs misattributed this check — the review's FLAW 4 is
  correct. `getIterationDomainTileFromOperandTiles` (`:212-231`) has *no*
  `isProjectedPermutation` check; the check is in the sibling
  `getIterationDomainTileFromResultTile` (`:261-288`). I re-verified both.]

So **no amount of contract editing makes the fill return a strided result
position.** The fill's result tensor is contiguous in its own coordinate space.
The `[2,1]` is not a property of the fill — it is a property of the
`tensor.insert_slice` that consumes the fill and writes it into a strided region
of the destination. That is the load-bearing observation.

### 1.2 The decisive question: which op's RESULT is the strided destination?

The writeback at `:616-617` is
`tensor::ParallelInsertSliceOp::create(…, tiledValue, destinationTensor,
resultOffset, resultSize, resultStride)` (`TileUsingInterface.cpp:619-621`). The
`resultOffset/resultSize/resultStride` describe **where the tile lands in the
destination tensor** — they come from `getResultTilePosition` of the *anchor op*
(the op being tiled). So the stride can only appear there if the anchor op's
result occupies a strided region of its destination.

- The fill's result = a freshly-filled contiguous tensor. Not strided.
- The `tensor.insert_slice`'s result = the destination *with a strided region
  overwritten*. **This is the strided result.**

Therefore the anchor must be the insert_slice, not the fill. The stride source
is the insert_slice's own `strides` attribute (`getMixedStrides()`), because
that attribute *is* the definition of how the source maps onto the strided
destination region. There is no other place the stride lives in the tensor IR.

This is what made me commit to mechanism (c): give `tensor.insert_slice` a
`TilingInterface` impl whose `getResultTilePosition` declares a strided result
position, and make it the tiled anchor.

### 1.3 Why this is NOT the refuted consumer-fusion mechanism

The refuted plan tried to fuse `tensor.insert_slice` as a *consumer* into a
tiled loop. That is refuted for three independent reasons (review §3): (1)
insert_slice is not `TilingInterface` today, so `tileAndFuseConsumer` rejects it
(`TileUsingInterface.cpp:2524`); (2) `transform.structured.fuse_into_containing_op`
does *producer* fusion, not consumer (`LinalgTransformOps.cpp:1335-1336`); (3)
even if reached, consumer fusion reads the loop-internal unit-stride candidate
and can only preserve, not create, a stride.

Mechanism (c) is the **opposite direction**: insert_slice is the **anchor**,
tiled *initially* by `tileUsingSCF`. Its result tensor IS the strided
destination, so reporting a strided result position is legitimate and
non-circular. There is no candidate-slice-preservation involved; the stride is
invented by the anchor's own `getResultTilePosition` from its own attribute.
This is the difference that makes the mechanism sound where the prior one was
unsound.

### 1.4 The only `TilingInterface` tensor op today is `tensor.pad` — the impl template

I confirmed the registration site: `TensorTilingInterfaceImpl.cpp:311-316`
attaches `TilingInterface` to `PadOp` only. The PadOp model
(`:24-84`) is therefore the closest precedent for an insert_slice model:
- `getLoopIteratorTypes` (`:26-31`): parallel × result-rank.
- `getIterationDomain` (`:33-44`): reified result shape.
- `getTiledImplementation` (`:46-55`): `bubbleUpPadSlice`.
- `getResultTilePosition` (`:57-66`): **identity** — `resultOffsets =
  offsets`, `resultSizes = sizes`. No stride.

The insert_slice model diverges from PadOp at exactly two points: the iteration
domain is the *source* shape (not the result/dest shape), and
`getResultTilePosition` computes a *strided* position (not identity). That
divergence is the whole point — but it is also the least-precedented part of the
plan (risk #1, §5 below).

---

## 2. The rejected alternatives — and the specific line that killed each

### 2.1 (a) Thread the destination stride as an INPUT to `getResultTilePosition`

**Idea:** add a `destStrides` *input* parameter to `getResultTilePosition`, so
the caller (SCF) hands the stride in, and the method uses it.

**What I checked:** the signature is `getResultTilePosition(b, resultNumber,
offsets, sizes, resultOffsets, resultSizes)` (`TilingInterface.td:118-162`). It
has *no destination operand*. The method is defined entirely in terms of the op
being tiled plus the iteration-space tile (`offsets`/`sizes`).

I then traced who *calls* it to see where a stride input could come from. The
sole caller on the initial-tiling path is the body lambda inside `tileUsingSCF`
(`TileUsingInterface.cpp:1211-1214`), which is built from the anchor op alone
plus `initTensors` (`createInitialTensorsForTiling:726`, which for a parallel op
calls `tensor::getOrCreateDestinations` at `:734`). There is **no "external
destination stride" anywhere in the tiling input** — `tileUsingSCF` (`:1112`)
takes `(rewriter, op, options)`. For SCF to obtain a stride to thread in, it
would have to invent a new caller-provided parameter, which is mechanism (b),
not a clean contract extension. The method is structurally incapable of
receiving a stride from outside without becoming (b).

**Killed by:** `TilingInterface.td:150-157` (the args list has no destination) +
`TileUsingInterface.cpp:1112` (`tileUsingSCF` has no stride input) +
`:1211-1214` (the caller builds from the op alone).

### 2.2 (b) A new "tile against a strided destination" tiling mode

**Idea:** a new SCF entry point `tileUsingSCFWithStridedDestination(op,
destStrides, …)` that tiles the fill but writes tiles into a caller-provided
strided destination.

**What I checked:** this would mean a new public API surface and a new parameter
on `generateLoopNestUsingForallOp` (`:556-563`), plus the caller (IREE) has to
*extract* the stride from the insert_slice and pass it in — duplicating the
slice op's geometry in a second place. It is a special-case "strided destination
mode" rather than a natural consequence of an op's own result position.

It is *viable* — it would work — but it maximizes SCF churn (a foundational file
the review flagged as "very high blast radius") and offloads the stride-source
problem to the caller rather than solving it at the op. It is strictly more
invasive than (c) for the same outcome. I kept it as a noted fallback (plan §11)
but rejected it as primary.

**Killed by (relative cost):** `TileUsingInterface.cpp:556-563` (would need a new
param) + the duplication argument + blast-radius comparison vs (c), which needs
only one new out-param on an existing method + one new implementor.

### 2.3 (d) A synthetic test-only op with a strided result position

**Idea:** add a `TestStridedWriteOp` implementing `TilingInterface` with a
strided `getResultTilePosition`, tile it, prove SCF emits the stride. Defer the
real insert_slice impl.

**What I checked:** this proves "SCF's plumbing can emit a stride" but does NOT
prove "`tensor.insert_slice` can be the strided anchor" — which is the
*production mechanism* and the whole reason the dispatch fails. The stride
genuinely lives on the insert_slice (`getMixedStrides()`); a stand-in leaves the
real topology unproven, so GO would not actually de-risk the IREE path.

The refuted plan had already dropped `Test_TilingNoDpsOp` for the same reason
("the stride source is the real candidate `tensor.insert_slice`, not a synthetic
test op" — `approach1_phase1_impl_plan.md:154-155`). That judgment is still
correct: the probe must exercise the real op to be load-bearing.

**Killed by:** the de-risk-value argument (a stand-in proves plumbing, not the
production mechanism) + the refuted plan's own prior conclusion.

---

## 3. The verified end-to-end call chain — `tile_using_forall` → `:616-617`

I traced this myself, opening each hop. Every line number below is current in
this checkout.

```
transform.structured.tile_using_forall            [LinalgTransformOps.cpp:3942  TileUsingForallOp::apply]
  │  for each target payload op:
  └─ tileToForallOpImpl(rewriter, state, transformOp, target, …)   [:3973]
       │  auto tileableOp = dyn_cast<TilingInterface>(target);     [:3899]
       │  if (!tileableOp) return emitSilenceableError(...);       [:3900-3903]  ← today's RED lives here
       │  scf::tileUsingSCF(rewriter, tileableOp, options);        [:3918-3919]
       └─ mlir::scf::tileUsingSCF(rewriter, op, options)           [TileUsingInterface.cpp:1112]
            │  iterationDomain = op.getIterationDomain(...)        [~:1119 region; impl-defined]
            │  innerYieldTiledValuesFn = GenerateTiledBodyFn lambda  [:1158-1226]
            │  initTensors = createInitialTensorsForTiling(...)    [:1229]
            │      → tensor::getOrCreateDestinations(op, …)        [:734]   → [insert_slice.dest]
            └─ generateLoopNest(rewriter, loc, options, iterationDomain,
                                givenTileSizes, numThreads, initTensors,
                                innerYieldTiledValuesFn)           [:1241]
                 │  (ForallOp branch):
                 └─ generateLoopNestUsingForallOp(rewriter, loc, loopRanges,
                        givenTileSizes, numThreads, mappingVector,
                        destinationTensors=initTensors, tiledBodyFn)  [:713]
                      │  forallOp = scf::ForallOp::create(…, outerDestinationTensors, …)  [:585-592]
                      │  offsets, sizes = getTileOffsetAndSizesWithForAllOp(
                      │      ivs, loopRanges, givenTileSizes, numThreads)   [:602]   (§4 numThreads math)
                      │  tiledBodyFn(rewriter, loc, ivs, offsets, sizes,
                      │      innerDestinationTensors, tiledResults,
                      │      resultOffsets, resultSizes)          [:607]
                      │     │  // (5c) tile the cloned anchor:
                      │     │  getTiledImplementation(clonedOp, offsets, sizes)  [:1194]
                      │     │     → %src_tile = tensor.extract_slice %source[offsets][sizes][1..]
                      │     │  // (5e) result position — THE CONTRACT SURFACE:
                      │     │  getResultTilePosition(op, index, tileOffsetsVec,
                      │     │      tileSizesVec, resultOffset, resultSize)      [:1211-1214]
                      │     │     ↑ after the contract change, also returns resultStrides
                      │     │     insert_slice impl:
                      │     │       resultOffsets[d] = base[d] + offsets[d]*str[d]  = [iv*2, 0]
                      │     │       resultSizes[d]   = sizes[d]*str[d]              = [2, 4]
                      │     │       resultStrides[d] = str[d]                       = [2, 1]
                      │     │     tiledResults = [%src_tile]
                      │  rewriter.setInsertionPointToEnd(forallOp terminator)      [:612]
                      │  for (tiledValue, dest, resultOffset, resultSize) in zip:  [:613-615]
                      │     SmallVector<OpFoldResult> resultStride(N,
                      │         rewriter.getIndexAttr(1));                         [:616-617]  ← HARDCODE TODAY
                      │     tensor::ParallelInsertSliceOp::create(rewriter, loc,
                      │         tiledValue, dest, resultOffset, resultSize,
                      │         resultStride);                                     [:619-621]
                      └─ return forall                                            [:623]
```

**The stride enters at exactly one place** — `getResultTilePosition` (the
`:1211-1214` call) — and travels two more hops to the writeback:
1. `getResultTilePosition` returns it (new `resultStrides` out-param).
2. The `GenerateTiledBodyFn` lambda carries it out via the new field on the
   typedef (`TileUsingInterface.cpp:359-364`), populating `resultStrides` in the
   `:1221-1222` region.
3. `generateLoopNestUsingForallOp:616-617` reads the channel value instead of
   `rewriter.getIndexAttr(1)`.

The `scf.for` twin is symmetric: `generateLoopNestUsingForOp:447-448` is the
identical hardcode, reached via the same `tiledBodyFn` (`:430`) → `:447`.

**RED confirmation (personally verified):** today, `tile_using_forall` on an
insert_slice dies at `tileToForallOpImpl`'s
`dyn_cast<TilingInterface>(target)` (`LinalgTransformOps.cpp:3899-3900`), which
emits a silenceable failure. So the Stage-1 RED is a clean "not TilingInterface"
rejection, not a copy-anchor rejection or a fusion-never-fired ambiguity. After
the insert_slice impl lands (Stage 3), the cast succeeds and the chain runs to
`:616-617`.

---

## 4. Stage/gate design rationale

### 4.1 Why propagate-first, flip-last (R1)

The unit-stride assumption is **systemic** — copy-pasted across the whole
structured-ops stack (feasibility doc §8.2). There are two *kinds* of site:
**hardcodes** that *drop* a stride (`:447-448`, `:616-617`, `:951-955`,
`:1006-1007`, `:1565-1566`, `:2366-2367` — all re-verified in this checkout) and
**rejections** that *gate* on it (`:2313-2317`, `:1502-1504`,
`SwapExtractSliceWithProducerPatterns.cpp:31-33` and `:99-101`).

The danger is asymmetric and directional: **removing a rejection before the
hardcodes can carry the stride produces a silent miscompile** — the candidate
passes the gate, then its strides are dropped at the writeback, writing to the
wrong memory. That is strictly worse than today's hard compile error. So the
invariant is: thread the channel (default unit → zero behavior change) across
*all* hardcodes FIRST; only then flip any rejection, and only the one a given
stage actually needs.

I verified the channel is *shareable* across all six hardcodes via the two
typedefs: `GenerateTiledBodyFn` (`:359-364`) feeds the initial-tiling pair
(`:447-448`, `:616-617`); `YieldTiledValuesFn` (`:336-340`) feeds the fusion
pair (`:951-955`, `:1006-1007`) and the dest-extract pair (`:1565-1566`,
`:2366-2367`). One field added to each typedef reaches all six. That is what
makes Stage 2 a single, behavior-neutral commit.

### 4.2 Why G3 (IR) + G4 (EXEC) is well-defined — and free of the prior ambiguity

The refuted plan's gate was ill-defined (review §5.5) because consumer fusion
could never fire for an insert_slice, so EXEC could not distinguish "SCF
composition wrong" from "fusion never ran" — the gate always read failure for
the wrong reason.

My gate has no such failure mode because **both halves are independently
achievable and checkable:**

- **G3 (IR) is producible:** the insert_slice impl (Stage 3) + the contract
  change (Stage 2) genuinely produce the strided writeback — I traced the chain
  in §3, and the RED is a clean TilingInterface-cast rejection that Stage 3
  fixes. There is no "fusion never fired" path; this is initial tiling.
- **G4 (EXEC) can run:** the strided `parallel_insert_slice` lowers through
  bufferization + LLVM (possibly *out-of-place* — see risk #2 — but it runs).

Crucially, G3 and G4 test **different things**, so failure modes are
distinguishable:
- G3 fails → the *mechanism/impl* is wrong (stop; do not proceed to G4).
- G3 holds, G4 wrong cells → silent miscompile in the offset/size/stride
  composition (exactly what R2 exists to catch; do not commit).
- G3 holds, COMPILE fails → the IR is correct but the *lowering stack* rejects
  the stride (record the next wall; the contract is still proven at IR level).

This three-way separation is what makes "NO-GO" informative rather than
ambiguous. The prior plan could not separate these because the fusion step was a
prerequisite it could not satisfy.

### 4.3 The offset math is correct *by construction* under numThreads

I was specifically asked to address the `iv*T*S` math under `useNumThreads`
(the effective tile is `ceilDiv(range, numThreads)`, not the literal size).
`getTileOffsetAndSizesWithForAllOp` computes the per-tile offset as
`loopRange.offset + iv * givenTileSize` (`offsetExpr = d0 + d1*s0`,
`TileUsingInterface.cpp:491`), and the comment at `:525-528` confirms
`givenTileSize` is the caller-computed `ceilDiv(range, numThreads)`. So
`offsets[d]` passed into `getResultTilePosition` is *already* the effective-tile
offset. The insert_slice impl computes `resultOffset[d] = base[d] +
offsets[d]*str[d]` — it never sees a "literal tileSize"; it sees the effective
offset value. The `iv*T*S` composition is therefore satisfied *for free*, with T
= the true effective size. The only discipline this imposes is a TEST that uses
a case where `ceilDiv(range,numThreads) != 1` (Stage 5.2), else the distinction
is invisible.

---

## 5. The two top risks — with the reasoning behind each

### Risk 1 [HIGHEST UNCERTAINTY]: the insert_slice impl's iteration-domain & size semantics

**What I'm asserting:** iteration domain = the *source* shape;
`getTiledImplementation` emits a contiguous `extract_slice` of the source (the
stride lives entirely in the writeback, never in the source read);
`getResultTilePosition` returns `resultOffsets[d] = base[d] +
offsets[d]*str[d]`, `resultSizes[d] = sizes[d]*str[d]` (the "span"),
`resultStrides[d] = str[d]`.

**Why I'm unsure:**
- `tensor.insert_slice` is **rank-reducing** and carries
  `OffsetSizeAndStrideOpInterface` subtleties; its "result" is the destination
  *with a strided overwrite* — an unusual anchor. The only tensor-dialect
  `TilingInterface` precedent is `tensor.pad` (`TensorTilingInterfaceImpl.cpp:24-84`),
  whose `getResultTilePosition` is pure identity. So the insert_slice impl has
  **no direct precedent** for "iteration domain = source, result position =
  strided."
- The `size = span` (`tileSize*stride`) vs `size = minimal`
  (`(tileSize-1)*stride+1`) choice is genuinely ambiguous for stride-2: both are
  stride-consistent (`ceilDiv(span,str) == ceilDiv(minimal,str) == tileSize` for
  the cases I checked) and both give non-overlapping regions for the probe's
  tile-divisible case. But the choice affects the "no overlap" contract
  (`TilingInterface.td:146-147`) for *non-tile-divisible* cases, and it affects
  what bufferization/LLVM will accept. I picked the span convention (consecutive
  tile regions `[iv*ts*str, (iv+1)*ts*str)` abut cleanly), but I cannot prove
  from the code that it is the *universally* correct one — only that it is
  correct for the static-2, tile-divisible probe.
- `[INFERENCE]` There may be a subtlety with `getOrCreateDestinations`
  (`TileUsingInterface.cpp:734`) returning the insert_slice's dest operand that I
  have not fully traced for the rank-reduced case.

**Why it's the top risk:** the impl design is the one place where I'm designing
new behavior rather than threading an existing value, and R2/G4 can only catch a
*wrong* choice *after* it's built — they cannot prove the design is right ahead
of time. This is the part most likely to surprise at implementation.

### Risk 2 [HIGH]: bufferization/lowering of the strided writeback may not survive to LLVM

**What I'm asserting:** G4's EXEC pipeline (plan §8) lowers the strided
`parallel_insert_slice` through one-shot bufferization + the `-convert-*` stack
to LLVM. I captured the pipeline shape from two Integration templates
(`test/Integration/Dialect/Vector/CPU/transfer-write.mlir:1-4` and the ArmSVE
`contraction.mlir` compile/run split) but did **not** validate it against
strided tensor IR.

**Why I'm unsure:** the unit-stride assumption is systemic below the tiling
layer too (feasibility doc §8.2). Specifically:
- `Tensor/Transforms/BufferizableOpInterfaceImpl.cpp:672-674` has an
  `allStridesOne` gate on `insert_slice` **in-place aliasing** — exactly the op
  my writeback produces. If it bails, bufferization falls back out-of-place
  (correct but slow) — G4 still holds. But `[INFERENCE]` if it fails *hard*
  rather than falling back, COMPILE errors and G4 is NO-GO-for-a-different-
  reason.
- `DataLayoutPropagation.cpp:1480-1483` rejects strided extract propagation;
  `Tensor/Transforms/ReshapePatterns.cpp:577` rejects strided reshapes. These
  may or may not be on the lowering path for the probe's specific IR.
- The vector/LLVM lowering of a strided scatter hits the *vectorization*-layer
  sibling of the same wall — LLVM #51660 (feasibility doc §8.1) — which
  degrades gracefully (vectorization disabled → scalar strided stores, correct
  but slow), so it should not block the EXEC, but I have not run it.

**Why it's high risk:** if any of these fail *hard*, Phase 0 delivers an
*IR-level* stride (G3 green) but not an *executable* one (G4 red). That is still
a useful result (it proves the contract change and localizes the next wall), but
it is a weaker go/no-go than a green G4, and the review should know the EXEC is
the less-certain half. The exact `-convert-*` sequence is a known unknown that
must be validated at implementation time; `-test-lower-to-llvm` (the monolithic
test pass used by `transfer-write.mlir:1`) is the fallback if the explicit
pipeline drops the stride.

---

## Appendix — the facts I re-verified that corrected the prior docs

(These are not reasoning; they are corrections that shaped the reasoning, recorded
so they are not re-litigated.)

- **File path correction:** `SwapExtractSliceWithProducerPatterns.cpp` is at
  `lib/Dialect/Tensor/Transforms/`, **not** `lib/Dialect/Linalg/Transforms/`
  (both the expert review and the feasibility doc cited the wrong directory). I
  confirmed via glob. The two guards are at `:31-33` and `:99-101` with the
  smoking-gun comment *"`TilingInterface` currently only supports strides being
  1."*
- **Projected-permutation check location:** the review's FLAW 4 is correct —
  `getIterationDomainTileFromOperandTiles` (`TilingInterfaceImpl.cpp:212-231`)
  has *no* `isProjectedPermutation` check; that check is in the sibling
  `getIterationDomainTileFromResultTile` (`:274`). `getMappedOffsetAndSize`
  (`:156-208`) enforces the property via `dyn_cast<AffineDimExpr>` at `:169-171`.
  This matters because it means stride-dividing *sizes* (data) does not touch
  the *map* (structure) — but it's moot for the probe anyway, since the fill is
  unit-stride and the stride is in the insert_slice anchor, not in any linalg
  indexing map.
- **All six hardcode + two rejection sites re-confirmed at the cited lines** in
  this checkout (`:447-448`, `:616-617`, `:951-955`, `:1006-1007`, `:1565-1566`,
  `:2366-2367`, `:2313-2317`, `:1502-1504`).
- **Feasibility doc §4.3 #7's stale claim** that "insert_slice *is*
  TilingInterface" is internally contradicted by the same doc's §2.1 correction
  (lines 91-100). It is false today; after this plan's Stage 3 it becomes true.
