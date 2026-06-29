# Expert Review — Approach 1, Phase 1 Implementation Plan (v3.1)

**Reviewer:** senior MLIR/LLVM/IREE compiler architect (adversarial second opinion).
**Subject:** `approach1_phase1_impl_plan.md` (DRAFT v3).
**Method:** every claim below cites a file:line the reviewer personally opened in
`~/Developer/iree` / `third_party/llvm-project`. `[INFERENCE]` marks reasoning not
directly observed. **No code was changed; no builds were run.**

---

## 1. Verdict — NO-GO as written (the SCF-only consumer-fusion bet is unsound)

**Recommendation: the `TilingInterface` contract change (or an equivalent
prerequisite) is NOT a go/no-go fallback — it (and a correct fusion topology) is
a prerequisite the plan cannot sidestep.**

The plan's thesis is that the `[2,1]` strides the SCF tiler needs are "already
read from the candidate slice op at `TileUsingInterface.cpp:2308-2311` and
discarded," so an SCF-only consumer-fusion probe can emit a strided writeback
without touching the contract. That thesis is built on a role confusion that
makes the plan unexecutable, and it is wrong in **three independent ways**, any
one of which is fatal:

1. **`tensor.insert_slice` does not implement `TilingInterface`** in this tree,
   so it cannot be fused as a consumer. `tileAndFuseConsumer` rejects it at
   `:2524`; `replaceInsertSlicesWithTiledConsumer` rejects it at
   `SwapExtractSliceWithProducerPatterns.cpp:77-80`. The cited consumer sites
   (`:2313`, `:2366`) are never reached.
2. **The specified harness (`transform.structured.fuse_into_containing_op`)
   performs producer fusion**, not consumer fusion. It calls
   `tileAndFuseFirstExtractUse` (`LinalgTransformOps.cpp:1335-1336`), which never
   touches `tileAndFuseConsumerOfSlicesImpl`.
3. **`:2311` reads the strides of the *candidate* (the loop-internal
   parallel_insert_slice), not the `[2,1]` consumer.** The SCF tiler always
   emits unit-stride candidates (`:616`, `:1006`), so consumer fusion can only
   *preserve* a stride that already exists in the candidate — it **cannot create**
   a strided writeback from a unit-stride one. The plan's goal is therefore
   architecturally unreachable via the consumer-fusion path.

This vindicates the feasibility doc's original verdict (`approach1_tensor_level_fusion.md`
§1, §2.4.1): the unit-stride assumption is *forced by the contract one layer up*,
and the SCF-only shortcut the plan bets on does not exist for this topology.

---

## 2. Confirmed claims (file:line verified)

These are correct and were re-opened:

- **Stride read-then-discard.** `TileUsingInterface.cpp:2309-2311` reads
  `getMixedOffsets/Sizes/Strides()`; `:2314` checks `all_of(strides, isOneInteger)`;
  `:2319-2320` forward only offsets/sizes into `allOffsets`/`allSizes`. Strides
  are dropped. ✓ (§1, R1 of the plan)
- **`:2330-2332`** calls `getIterationDomainTileFromOperandTiles(... allOffsets,
  allSizes ...)`. `allSizes` is the candidate's raw `getMixedSizes()` — un-divided. ✓
- **`:2347-2349`** calls `getResultTilePosition`. ✓
- **`:2366-2367`** dest-extract hardcodes `getIndexAttr(1)` strides. ✓
- **`:1006-1007`** (forall fusion writeback) hardcodes `getIndexAttr(1)`. ✓
- **`:951-955`** (scf.for fusion writeback), **`:616-617`** (forall initial
  writeback) hardcode `getIndexAttr(1)`. ✓
- **`:1502-1504`** producer-fusion stride rejection; **`:1565-1566`** producer
  dest-extract hardcodes `1`. ✓
- **Initial tiling `:556-624`** calls `tiledBodyFn` (`:607`) and emits only the
  writeback (`:616-621`); there is **no dest-extract** in `generateLoopNestUsingForallOp`
  itself. ✓ (the v3 "site reachability" correction is right)
- **`tensor.insert_slice` IS `DestinationStyleOpInterface`**
  (`TensorOps.td:843`). ✓ — but see §3.1: DPS does not save the plan.
- **`YieldTiledValuesFn` typedef** at `:336-340` (shared by the fusion sites). ✓

So the *citation-level* facts in the plan are accurate. The problem is what they
are claimed to *mean* for the chosen topology (§3).

---

## 3. Critical flaws / showstoppers (ranked by severity)

### FLAW 1 (FATAL) — `tensor.insert_slice` is not `TilingInterface`; it cannot be the fused consumer

Evidence, all personally opened:

- The trait list for `InsertSliceOp` (`TensorOps.td:838-848`) contains
  `DestinationStyleOpInterface`, `OffsetSizeAndStrideOpInterface`, etc., but **no
  `TilingInterface`**.
- The tensor-dialect TilingInterface registration
  (`lib/Dialect/Tensor/IR/TensorTilingInterfaceImpl.cpp:311-316`) attaches the
  interface **only to `tensor::PadOp`**. `insert_slice`/`extract_slice` get
  `SubsetOpInterface` (`SubsetInsertionOpInterfaceImpl.cpp:92-96`),
  `BufferizableOpInterface` (`BufferizableOpInterfaceImpl.cpp:1207`),
  `TiedOpInterface` (IREE `UtilExternalModels.cpp:1386`), etc. — **never
  `TilingInterface`** (confirmed across all of `third_party/llvm-project/mlir` and
  `compiler/src`).

Consequences for the plan:

- `tileAndFuseConsumer(rewriter, consumer, loops)` checks
  `isa<TilingInterface>(consumer)` at `:2524-2527` and returns
  *"unhandled consumer that does not implement TilingInterface"* for an
  `insert_slice`. The call never reaches `tileAndFuseConsumerOfSlicesImpl`, so
  `:2313` (stride rejection), `:2347`, `:2366` are **never executed** for a §3.1
  consumer.
- The alternate entry `tileAndFuseConsumerOfSlices` (`:2421`) does reach
  `tileAndFuseConsumerOfSlicesImpl`, but `replaceInsertSlicesWithTiledConsumer`
  (`SwapExtractSliceWithProducerPatterns.cpp:62`) does
  `dyn_cast<TilingInterface>(consumerOperands.front()->getOwner())` at `:77-80`
  and `return failure()` for a non-TilingInterface consumer. So the consumer
  **cannot be tiled**, and fusion bails at `:2286-2288`.

**This refutes the feasibility doc's claim** (`approach1_tensor_level_fusion.md`
§2.1) that *"the strided `tensor.insert_slice` … is a `TilingInterface` op and
therefore is a `computeOp`."* It is neither; `isComputeOp`
(`Utils.cpp:980-982`) requires `TilingInterface | UKernelOpInterface`, and
`insert_slice` has neither. The plan inherited this error.

The plan's §3 narrative ("both §3.1 and §3.2 reach BOTH `:1006` and `:2366`
because `insert_slice` is DPS") is therefore irrelevant: the DPS check at
`:2224`/`:2358` would succeed, but the fusion dies earlier at `:2524` (or at
`replaceInsertSlicesWithTiledConsumer`). You never get to exercise DPS.

### FLAW 2 (FATAL) — The specified harness does producer fusion, not consumer fusion

The plan repeatedly says to drive the test with
`transform.structured.fuse_into_containing_op` on the `tensor.insert_slice`
(§3 header, §A.1). That is not what this transform does.

- `FuseIntoContainingOp::apply` (`LinalgTransformOps.cpp:1271-1383`) fuses a
  **producer** into a **containing** op. Its body calls
  `tileAndFuseFirstExtractUse(rewriter, …, producerOp, containingOp)` at
  `:1335-1336`, then `tileAndFuseFirstExtractUseThroughContainingOpBlockArgument`
  (`:1360-1362`), then `cloneAndFuseFirstUse` (`:1370-1371`). **None of these
  calls `tileAndFuseConsumer` or `tileAndFuseConsumerOfSlicesImpl`.**
- `tileAndFuseFirstExtractUse` (`:992-1107`) searches for an
  `extract_slice` *user* of the producer inside the containing op (`:1006-1009`)
  and tiles the producer via `generateResultTileValue` (`:1068-1070`). This is
  **producer** fusion (compute the producer's slice inside the loop), the
  opposite direction from what the plan describes.

Who actually drives the consumer-fusion code the plan wants to exercise? Only:
- `TestFuseConsumerOp` → `scf::tileAndFuseConsumer` (`test/lib/Interfaces/TilingInterface/TestTilingInterfaceTransformOps.cpp:184-185`)
- `TestFuseConsumerUsingSliceOp` → `scf::tileAndFuseConsumerOfSlices` (`:248-249`)
- IREE `fuseConsumersIntoForall` (`compiler/src/iree/compiler/Codegen/Common/TileAndFuseUtils.cpp:215-216`) and `Transforms.cpp:106-107`
- IREE `CommonExtensions.cpp:1155-1156`

So the plan would need `transform.test.fuse_consumer` (a test-only op) or an IREE
pass — and even then FLAW 1 still blocks the `insert_slice` consumer. The
canonical consumer-fusion test confirms the intended topology: in
`test/Interfaces/TilingInterface/tile-and-fuse-consumer.mlir:5-37` the fused
**consumer is `linalg.add`** (a TilingInterface op) and the **candidate is the
unit-stride `tensor.insert_slice` inside the loop** — exactly the opposite of the
plan's setup.

### FLAW 3 (FATAL / ARCHITECTURAL) — Role confusion: candidate strides ≠ consumer strides; consumer fusion cannot *create* a strided writeback

Even fixing FLAW 1 and FLAW 2, the central mechanism is circular.

Trace of `tileAndFuseConsumer` (`:2521-2570`):

- The **candidate slices** are obtained from `getProducingInsertSliceLikeOp`
  (`:2558-2566`). For a `scf.forall`, that is `getProducingParallelInsertSlice`
  (`:2487`) — i.e. the **parallel_insert_slice inside the loop** that produces
  the loop result.
- These candidates are cloned (`cloneAsInsertSlices`, `:2258-2259`) and their
  offsets/sizes/**strides** are read at `:2309-2311`.

**The candidate is produced by the SCF tiler**, which hardcodes unit strides at
`:616`/`:1006`. So `candidateSliceOp.getMixedStrides()` at `:2311` returns `[1,1]`
in every SCF-tiler-produced loop. The `[2,1]` the plan wants lives on the
**consumer** `insert_slice` (`%r = tensor.insert_slice %filled into %dest[0,0][4,4][2,1]`),
and **the consumer's strides are never read anywhere in
`tileAndFuseConsumerOfSlicesImpl`** — I traced every use of `consumerOp` /
`clonedConsumerOp`; its offsets/sizes/strides are baked into the cloned consumer
and consumed by `replaceInsertSlicesWithTiledConsumer`, never threaded to
`:2311`/`:2330`/`:2347`.

Therefore the plan's Step A.2(c) instruction — *"keep candidate strides
(`:2311`); stride-divide `allSizes` …; thread strides to all four sites"* —
threads **`[1,1]`** through the system. It changes nothing. The expected §3.1-B
output (`parallel_insert_slice … [2,1]`) cannot be produced because there is no
`[2,1]` entering the consumer-fusion path.

More fundamentally: **consumer fusion *preserves* the candidate's writeback
geometry; it cannot transform a unit-stride candidate into a strided writeback.**
The only place a stride could be *introduced* is **initial tiling**
(`:616`/`:447`), when the *anchor op itself* has a strided result-to-dest
relationship — which is precisely where the plan *defers* and precisely where the
`getResultTilePosition` contract (returning offsets/sizes only) forces unit
strides. That loops straight back to the contract change the plan demotes to
fallback.

### FLAW 4 (CITATION ERROR) — The projected-permutation check is misattributed

The plan (R2, §9 Q2, and the shared context) states
`getIterationDomainTileFromOperandTiles`'s "projected-permutation assumption" is
at `TilingInterfaceImpl.cpp:268-275`. It is not.

- `getIterationDomainTileFromOperandTiles` is `TilingInterfaceImpl.cpp:212-231`.
  It builds the indexing maps (`:220-224`) and calls `getMappedOffsetAndSize`
  (`:225`). **It has no `isProjectedPermutation()` check.**
- The `isProjectedPermutation()` check at `:268-278` (the cited `:268-275`) lives
  in a **different method**, `getIterationDomainTileFromResultTile` (`:261-288`).
- `getMappedOffsetAndSize` (`:156-208`) *does* enforce a projected-permutation-like
  property, but via `dyn_cast<AffineDimExpr>` on each map result at `:169-171`
  (`return failure()` if not a dim expr) — not via the named check.

Practical implication for the plan's §2 worry ("does stride-division preserve the
permutation property?"): **moot.** Stride-dividing the *sizes* (data values) does
not touch the indexing *map* (structure), so the `AffineDimExpr` requirement is
unaffected. For a `linalg.fill` (identity map) the operand-tile method is a pure
passthrough (`iterDomainSizes ← allSizes`), so feeding stride-divided sizes would
indeed yield a smaller iteration domain — but you never reach `:2330` for an
`insert_slice` consumer (FLAW 1), and the candidate is unit-stride anyway (FLAW 3).

### FLAW 5 (INVENTORY GAP) — Two real stride-rejection sites are missing from the "6 + 2" inventory

The plan's inventory lists six hardcoded-`1` sites and two rejections, all in
`TileUsingInterface.cpp`. The actual execution paths also hit two **additional**
unit-stride guards in the tensor dialect, with the smoking-gun comment the
feasibility doc itself quoted:

- `SwapExtractSliceWithProducerPatterns.cpp:31-33` (`replaceExtractSliceWithTiledProducer`):
  `// TilingInterface currently only supports strides being 1.` →
  `if (!llvm::all_of(sliceOp.getMixedStrides(), isOneInteger)) return failure();`
- `SwapExtractSliceWithProducerPatterns.cpp:99-101` (`replaceInsertSlicesWithTiledConsumer`):
  the identical guard on the *candidate* slices — this is the function called at
  `TileUsingInterface.cpp:2284`, i.e. on the consumer-fusion path the plan targets.

These are not academic: `replaceInsertSlicesWithTiledConsumer:99-101` sits on the
very path the plan's §3.1/§3.2 would take (if they could take it). Any
"thread strides through SCF" change that ignores these two will still fail here.

---

## 4. The hand-derived IR (§3.1, §3.2) — correct in the abstract, moot in practice

Independently deriving the expected post-fusion IR:

**§3.1-B (tile-size-1, stride-2, 2 tiles into a 4-row dest).** Source `tensor<2x4>`,
stride-2 on dim-0. Insert-slice semantics: `source_size[d] = ceilDiv(sizes[d],
strides[d])`, so `sizes[0] = source_size[0]*stride = 1*2 = 2`. Per tile `iv`:
- dest offset of the single source row = `iv*stride = iv*2`; ✓
- the per-tile writeback is `parallel_insert_slice %tile into %dest[iv*2,0][2,4][2,1]`. ✓

The plan's `iv*2` is **correct**, and `iv*tileSize*stride` (general form `base +
iv*T*S`) is the right composition *for an identity-map op like fill*. (For a
non-trivial indexing map, post-multiplying the *result* offset by the stride is
only valid on the dims that are strided in the destination space; the plan does
not address this, but it is scoped to fill, so acceptable.)

**§3.2 (½× read, `[1,2]`).** Dest-extract of the tiled DPS init carrying `[1,2]`
on dim-1 is the correct expectation for that case. ✓

**R2 miscompile-detection is sound.** If the writeback were
`…into %dest[iv,0][2,4][2,1]` (offset `iv`, not `iv*2`): tile 0 → dest row 0
(happens to be right), tile 1 → dest row 1 (**wrong**, should be row 2). A
FileCheck matching only `[2,1]` would false-pass; the offset is exactly where the
silent miscompile hides. Asserting offset **and** sizes **and** strides is the
correct discipline. R2 is the strongest part of the plan and should survive into
whatever approach replaces this one.

**Caveat:** all of §3 is unreachable via the plan's mechanism (FLAW 1–3). The
hand-derived IR proves the *target* is well-defined; it does not prove the plan
can hit it.

---

## 5. Understated risks the plan glosses over

1. **Consumer fusion cannot manufacture strides (architectural), not just a
   wiring gap.** Even with infinite SCF edits, the candidate is unit-stride, so
   `:2311` has no stride to propagate. The plan frames this as "thread what's
   already there"; there is nothing there.

2. **The original dispatch is dynamic; the probe is static.** `m[0::2,0::2]=True`
   is `tensor<?x?xi8>` (feasibility doc §1). Stride-dividing a *dynamic* span
   requires runtime `ceildivi` and the iteration domain becomes dynamic — the
   static `tensor<4x4>` probe would not transfer. The plan scopes to "static
   stride 2" without noting this does not de-risk the actual failure.

3. **The original failure also needs IREE filter widening + co-distribution**
   (plan §8, §9). Even a hypothetical working SCF strided writeback would not, by
   itself, compile `m[0::2,0::2]=True`: the transposed stride mismatch
   (`[1,2]` region vs `[2,1]` output) that forces the bufferization
   `storage_buffer` copy is upstream of any single writeback. Phase 0 proving
   "SCF can emit a stride" would still leave the dispatch broken.

4. **The EXEC gate (R3/A.4) cannot be set up.** `mlir-cpu-runner` exercises
   lowered MLIR through the JIT; to exercise *fusion* you need
   `--transform-interpreter` (to fire `test.fuse_consumer`) **then** one-shot
   bufferization **then** lowering. The plan captures neither the transform
   sequence nor the bufferization pipeline (§9 Q3 defers it). And per FLAW 1–2,
   the fusion never fires, so there is no strided IR to execute regardless.

5. **The §6 fallback trigger is ill-defined.** *"EXEC fails proving SCF-local
   insufficient"* presupposes an EXEC that can run and a strided IR that can
   miscompile. Because the fusion cannot fire, EXEC cannot distinguish "SCF
   composition wrong" from "fusion never ran." The gate would always read
   failure-for-the-wrong-reason — risking either a spurious "go to contract"
   (correct conclusion, wrong evidence) or, worse, an implementer "fixing" the
   harness until something green appears.

6. **`numThreads` vs `tileSizes` in `scf.forall`.** When tiling by `numThreads`
   (`getTileOffsetAndSizesWithForAllOp`, the `useNumThreads` branch at `:574`),
   the effective tile size is `ceilDiv(range, numThreads)`, not the literal tile
   size. Stride post-multiply math (`iv*T*S`) assumes `T` is the true per-tile
   size; under `numThreads` tiling this needs recomputation. The plan's tests use
   explicit tile sizes and never exercise the `numThreads` path that IREE actually
   uses for workgroup distribution.

---

## 6. Recommendations — smallest de-risking set

1. **Stop treating the contract change as a fallback.** For the goal stated
   (create a strided tensor writeback for a scatter-fill), it is a prerequisite.
   The feasibility doc had this right; the v2/v3 "SCF-only" reframe is the error.
   Either commit to the contract change (`getResultTilePosition` returns strides,
   populate the five implementors), or pursue Approach 2/3 (the parked peers) /
   a local IREE fork.

2. **If you still want a cheap upstream probe, change *both* the topology and the
   harness.** Use a consumer that **is** `TilingInterface` (a `linalg` op, as in
   `tile-and-fuse-consumer.mlir:5-37`) and drive it with
   `transform.test.fuse_consumer`. But understand this probes *"can consumer
   fusion preserve a hand-written strided candidate"* — it does **not** probe
   *"can SCF create a strided writeback,"* which lives at `:616`/`:447 (initial
   tiling) and is contract-gated. State that distinction explicitly so a green
   result is not over-read.

3. **Fix the inventory to include the tensor-dialect guards**
   (`SwapExtractSliceWithProducerPatterns.cpp:31-33`, `:99-101`). They are on the
   real paths and must be changed in lockstep with the SCF sites (or the SCF
   change is dead on arrival).

4. **Preserve R2 and the §3.1-B/§3.2 hand-derived IR.** They are correct and
   portable to whichever approach is taken; they should become the acceptance
   contract for the eventual fix.

5. **Make the EXEC gate honest.** Specify the full pipeline
   (`--transform-interpreter` → `--one-shot-bufferize` → `--convert-*` → runner)
   from an existing `mlir/test/Integration/` template *before* relying on it as
   the §6 decision signal.

6. **Re-derive the producer-fusion angle explicitly** (what
   `fuse_into_containing_op` actually does). The strided `extract_slice` of a
   producer (`%sub = extract_slice %dest[…][1,2]`) feeding a fill is the natural
   producer-fusion case, and it hits `replaceExtractSliceWithTiledProducer:31-33`
   and `:1502`. That path is real and was conflated with consumer fusion here.

---

## 7. Answer to the two open decisions

**(a) Tighten one more pass (resolve the `mlir-cpu-runner` template + sub-case-B
site attribution) vs (b) Start Phase 0 execution now.**

**Recommendation: (a), emphatically — but go further than the plan frames it.**
The plan treats this as a 30-minute template-capture task. It is not. The
source reading above (≈the work of "one more pass") surfaced **three fatal
flaws** that execution would hit on the *first* `mlir-opt` run:

- Step 0.3 / A.1's "RED evidence" (`:2313-2317 rejects`) would **not** reproduce:
  the rejection actually fires at `:2524` ("does not implement TilingInterface")
  — a different op, different line, different reason — because the consumer is an
  `insert_slice`. Worse, with the specified `fuse_into_containing_op` harness the
  rejection is at `replaceExtractSliceWithTiledProducer:31-33` (producer path),
  not in `TileUsingInterface.cpp` at all.
- Even after "fixing" the harness to `test.fuse_consumer`, A.3's flip of `:2313`
  would produce **no** `:2366`/`:1006` traffic, because the candidate is
  unit-stride and the consumer cannot be tiled. RED would stay RED for the wrong
  reason, and GA could never go green.

Starting execution now burns a full build + a day chasing a RED that never
matches the predicted site, then a GREEN that can never appear. One more pass —
specifically: (i) re-derive §3 with a `TilingInterface` consumer and the correct
transform op, (ii) decide whether the probe is "preserve a strided candidate"
(contract-light, but circular for the goal) or "create a stride at `:616`"
(contract-required), and (iii) capture the real EXEC pipeline — converts a
multi-week dead end into an informed fork in the road.

The single most important correction: **the `[2,1]` stride is read from the
consumer `insert_slice`, which `:2311` never sees, because `:2311` reads the
candidate `parallel_insert_slice` (always `[1,1]`).** That one fact — verified at
`:2308-2311` + `getProducingParallelInsertSlice:2487` + the tiler's hardcoded
`:616`/`:1006` — is why the SCF-only bet cannot work for this topology and why
the contract change is the prerequisite, not the fallback.
