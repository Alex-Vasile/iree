# Approach 1 (CONTRACT) Plan — Adversarial Review (R1)

> Reviewer: senior MLIR/IREE compiler architect (adversarial pass).
> Target: `approach1_contract_phase1_plan.md` (v1). Mechanism under review: make
> `tensor.insert_slice` a `TilingInterface` anchor, introduce strides via a new
> `resultStrides` out-param on `getResultTilePosition`, flow to the `:616-617`
> writeback.
> Method: every `file:line` below was opened in this session. `[INFERENCE]` marks
> reasoning-only claims. No code changed, no builds run.

---

## 1. Verdict — FIXABLE-WITH-SPECIFIC-CHANGES (conditional GO)

A1's **mechanism is viable** — initial-tiling of `tensor.insert_slice` with a
strided `getResultTilePosition` is a coherent, reachable, non-circular topology,
and the call chain it depends on is **real** (§2). The "stride source is the
slice op's own geometry" thesis is correct and survives the refuted plan's three
flaws.

**But A1's central size formula is inverted**, and the inversion is not a
pinning choice — it produces **IR that cannot verify** in two independent places:
the canonical example (§3 F1) and the emitted per-tile writeback (§3 F2). A1 has
applied *extract_slice* semantics (`sizes` = dest span) to an *insert_slice*
op (`sizes` = source element count). The fix is mechanical (`resultSizes =
iterSizes`, not `iterSizes*stride`; Case-1 IR `sizes` `[2,4]` not `[4,4]`), and
with it G3 produces valid IR. The no-overlap contract is **satisfied** under the
corrected convention (§3 F1-fix analysis), so the mechanism is not killed.

Remaining risks after the fix: (a) the **blast radius** of giving `insert_slice`
`TilingInterface` is large and A1 under-mitigates it (§4) — it does not block
G3/G4 (which is a pure transform-interpreter test) but it lands squarely on the
IREE integration that the GO criterion authorizes; (b) **EXEC survival** is
genuinely untraced (§3 F4), A1's own risk #2, correctly flagged.

**Conditional GO:** GO if A2 adopts the `resultSizes = iterSizes` correction,
re-derives the Case-1 IR with `sizes = source shape`, and adds a blast-radius
containment note. NO-GO only if A2 cannot defend the corrected size math or the
EXEC pipeline hard-fails at bufferization.

---

## 2. Call-chain verification — CONFIRMED (every hop opened)

The chain A1 asserts is real. Traced personally, line by line:

| Hop | Cited site | Verified | Note |
|---|---|---|---|
| transform entry | `LinalgTransformOps.cpp:3942` `TileUsingForallOp::apply` | ✅ | loops payload ops, calls `tileToForallOpImpl` at `:3973` |
| TilingInterface gate | `:3899` `dyn_cast<TilingInterface>(target)` | ✅ | `:3900-3906` emits silenceable "only TilingInterface ops are supported" — this **is** today's clean RED (confirmed) |
| SCF entry | `:3919` `scf::tileUsingSCF(...)` | ✅ | options set `ForallOp` loop type at `:3909` |
| iteration domain | `TileUsingInterface.cpp:1122` `op.getIterationDomain(...)` | ✅ | drives loop bounds |
| body lambda | `:1158-1226` `innerYieldTiledValuesFn` (`GenerateTiledBodyFn`) | ✅ | clones op `:1178`, `getTiledImplementation` `:1194`, `getResultTilePosition` `:1211-1214` |
| init tensors | `:1229` `createInitialTensorsForTiling` → `getOrCreateDestinations` | ✅ | becomes forall `shared_outs` |
| loop nest | `:1241` `generateLoopNest(...)` | ✅ | dispatches to forall variant |
| forall body | `generateLoopNestUsingForallOp:556-624` | ✅ | computes iter offsets/sizes `:602`, calls `tiledBodyFn` `:607-609` |
| **writeback (THE site)** | **`:616-617`** | ✅ | `resultStride` is a **local** `SmallVector` hardcoded `rewriter.getIndexAttr(1)`, read at `:619-621` by `ParallelInsertSliceOp::create` |

**Channel question — answered YES.** `:359` (`GenerateTiledBodyFn`) IS the
channel that `:616` reads *indirectly*: the lambda populates `resultOffsets` /
`resultSizes` (out-params on the typedef, `:362-364`), which `:607-609` captures
and `:613-615` zips into the `:616` loop. `resultStride` today is **not** on that
channel — it is a fresh local at `:616-617`. A1's claim that you must (i) add a
`resultStrides` field to `GenerateTiledBodyFn` (`:359-364`), (ii) populate it in
the lambda (`:1221-1222`), and (iii) read it at `:616-617` is **exactly correct**
— this is the precise, minimal change site.

The static helper `getResultTilePosition` (`:848-874`) under
`FullReduction` (the default for a plain insert_slice, which is not a
`PartialReductionOpInterface`) is a clean passthrough to
`op.getResultTilePosition(...)` at `:859-860` — so threading a `resultStrides`
param here is mechanical. ✅

The RED A1 describes for Stage 1.2 is accurate: today `insert_slice` is not
registered (`TensorTilingInterfaceImpl.cpp:314` registers only `PadOp`), so
`:3899-3906` rejects it cleanly. ✅

---

## 3. Critical flaws (ranked by severity)

### F1 — CRITICAL, blocks G3. `resultSizes = iterSizes*stride` is semantically wrong → emitted writeback fails verification.

A1's §1.3 / §3 math: `resultSizes[d] = iterSizes[d] * strides[d]` (the "span").
For tile-1, stride-2 this yields `resultSizes=[2,4]`, and A1's G3 writeback is:

```mlir
tensor.parallel_insert_slice %tile into %dest[iv*2, 0] [2, 4] [2, 1]
  : tensor<1x4xi32> into tensor<4x4xi32>
```

**This IR is invalid.** `ParallelInsertSliceOp::verify`
(`TensorOps.cpp:3960-3982`) calls `verifyInsertSliceOp`
(`TensorOps.cpp:2885-2896`), whose core is:

```cpp
RankedTensorType expected =
    ExtractSliceOp::inferResultType(dstType, staticSizes);  // :2891-2892
return isRankReducedType(expected, srcType);                // :2895
```

i.e. **`sizes` define the expected SOURCE shape** (`inferResultType(dest, sizes)`
is the source type), and the actual source must match (rank-reduction aside).
Here `inferResultType(dest<4x4>, sizes[2,4])` = `tensor<2x4>`, but the source
tile is `tensor<1x4>` → `isRankReducedType(tensor<2x4>, tensor<1x4>)` **fails**
(2 ≠ 1, not a unit-dim drop). The op is rejected by the verifier before any
FileCheck can match. G3 is unreachable with A1's math.

**Root cause:** A1 has applied *extract_slice* reasoning. For `extract_slice`,
`sizes` is the span read out of the source and the result is the dilated region.
For `insert_slice`/`parallel_insert_slice`, **`sizes` is the number of source
elements placed**; the stride governs dest spacing, and the dest span is
`(sizes-1)*stride+1` (enforced in-bounds by `verifyInBoundsSlice` at
`:2910-2912` / `:3975-3977`). A1 even states the inverted mental model in §3:
*"span 4/stride2 = 2 source rows"* — that is the extract direction; insert is
the inverse.

**FIX (one line in the impl):** `resultSizes[d] = iterSizes[d]` (the per-tile
source size), keep `resultStrides[d] = insert.getMixedStrides()[d]`. Corrected
writeback:

```mlir
tensor.parallel_insert_slice %tile into %dest[iv*2, 0] [1, 4] [2, 1]
  : tensor<1x4xi32> into tensor<4x4xi32>
```

`inferResultType(dest<4x4>, [1,4])` = `tensor<1x4>` == source tile ✓; span
`(1-1)*2+1=1`, `offset iv*2` ∈ {0,2} → in-bounds ✓. **Valid.**

**No-overlap contract (TilingInterface.td:146-147) — RESOLVED, not a killer.**
Under the corrected convention, consecutive tiles write dest point-sets
`{iv·T·S + j·S : j∈[0,T)}`; tile `iv` and `iv+1` are separated by a gap of
`T·(S−1) ≥ 0`, so they are disjoint for any stride ≥ 1, tile ≥ 1. The contiguous
*box* descriptors (`resultOffset`,`resultSize`) are likewise disjoint. So the
contract holds — but **only because** `resultSizes = iterSizes`; A1's
`*stride` "span" variant is the one that would risk both contract violation AND
out-of-bounds. A1's §11 risk #1 worry ("if the span convention produces
overlapping regions") is moot: the span convention is simply wrong and must be
dropped.

### F2 — CRITICAL, blocks G3. The canonical Case-1 IR itself is invalid.

A1 §3 IR:
```mlir
%r = tensor.insert_slice %filled into %dest[0, 0] [4, 4] [2, 1]
      : tensor<2x4xi32> into tensor<4x4xi32>
```
`verifyInsertSliceOp` (`:2885-2896`): `inferResultType(dest<4x4>, sizes[4,4])` =
`tensor<4x4>`; source is `tensor<2x4>` → fail (dim-0: 4 ≠ 2, not rank-reducible).
Independently, `verifyInBoundsSlice` (`:2910-2912`): span dim-0 =
`(4-1)*2+1 = 7 > 4` → out-of-bounds. **Doubly invalid.** The transform-interpreter
test cannot even parse/verify, so the RED in Stage 1.1 is "IR rejected," not
"insert_slice not TilingInterface."

**FIX:** `sizes` must equal the source shape:
```mlir
%r = tensor.insert_slice %filled into %dest[0, 0] [2, 4] [2, 1]
      : tensor<2x4xi32> into tensor<4x4xi32>
```
`inferResultType(dest<4x4>, [2,4])` = `tensor<2x4>` ✓; span dim-0 = `(2-1)*2+1 =
3 ≤ 4` ✓. Writes dest rows `{0,2}` — the intended 2× dilation. (F1 and F2 are the
same inversion expressed in two places; fix both together.)

### F3 — HIGH. Blast radius of giving `insert_slice` `TilingInterface` is large and under-mitigated (§4). Not a G3/G4 blocker; lands on the GO-authorized integration.

See §4. Giving a dialect op an interface is a **global** change. ~30
`dyn_cast<TilingInterface>` / `isa<TilingInterface>` sites in MLIR + IREE would
newly match `insert_slice`, including IREE's own greedy GPU distributor
(`GPUGreedilyDistributeToThreads.cpp:139`) which is in the *exact* pipeline the
real `m[0::2,0::2]=True` bug lives in. A1's R1 invariant guards the *channel
threading*, not this. It does not block the transform-only G3/G4 probe, but the
GO criterion explicitly authorizes proceeding to that integration, so the
containment plan must be in the rework, not deferred vaguely.

### F4 — MEDIUM/HIGH. EXEC survival untraced; the bufferization path differs from A1's cited gate.

A1's risk #2 cites `BufferizableOpInterfaceImpl.cpp:672-674`
(`insertSliceOpRequiresRead`, the `allStridesOne` gate). That gate is on
**`tensor::InsertSliceOp`** (`:637-675`) and governs whether the *dest is read*
(in-place aliasing). It is **not** on the G3 emission path, which produces a
**`parallel_insert_slice` inside `scf.forall`** (`:619-621`).

`parallel_insert_slice` is bufferized **via the parent `scf.forall`**, not as a
standalone op — `BufferizableOpInterfaceImpl.cpp:1210-1212` (SCF) states the
terminator interfaces are "only used during analysis. Not for bufferization,"
and `ForallOpInterface::bufferize` (`:1243-1296`) does **not inspect strides**;
it replaces `shared_out` bbargs with `to_tensor(memref)` and `mergeBlocks`. So
the stride is not rejected at the forall bufferization. **[INFERENCE]** The
strided `parallel_insert_slice` then has to lower to a strided `memref.subview` +
copy, whose survival through `-convert-scf-to-cf` / `-convert-cf-to-llvm` /
LLVM is the real, unverified risk.

The smoking gun is `BufferizableOpInterfaceImpl.cpp:694-698` (the standalone
`InsertSliceOp::bufferize` author's own comment): *"insert_slice ops arise from
tiling and bufferizing them out-of-place is generally a deal breaker … cloning
the whole tensor on every single iteration … catastrophically bad scheduling
decision. TODO: be very loud about it or even consider failing the pass."* The
ecosystem's stance on tiled insert-slices is hostile. For the in-place
`parallel_insert_slice`-on-shared-memref form the situation is better, but A1
hasn't traced it, so G4 is the only mitigation and it is binary (pass/fail),
not explanatory. The first pass that could reject is downstream of forall
bufferization (memref strided-store lowering), not the cited `:672-674` gate.

**Mitigation A1 already has and should keep:** G4's three NO-GO modes (§10)
correctly separate "wrong cells" (math bug, F1/F2) from "compile fail" (lowering
rejection). With F1/F2 fixed, "wrong cells" becomes a real signal rather than a
dupe of invalid-IR.

### F5 — LOW (deferred, A1 acknowledges). Rank-reduction + dynamic sizes.

`insert_slice` is rank-reducing (`TensorOps.td:878-890`, `getDroppedDims`
`TensorOps.cpp:3217-3219`). A1's `getIterationDomain` returns the **source** rank,
but `getResultTilePosition` must return **dest**-rank vectors (the forall
`shared_out` is dest-typed). PadOp (`TensorTilingInterfaceImpl.cpp:24-66`) never
faces this — its iter domain = result rank (`:33-44`), an identity map. So A1 is
inventing not just strided semantics but a **rank-mismatch between iteration
domain and result position**, with no precedent. Fine to defer to a stretch, but
the rework must state the rank-expansion rule (drop-dims get size 1 / offset =
base, per `getDroppedDims`) explicitly before claiming the impl is "modeled on
PadOp." Dynamic sizes (`tensor<?x?x>`) need runtime `ceildivi` in the offset
composition — A1 notes this (§7, §9); the static probe does not transfer.

---

## 4. Blast radius — consumer sites that newly match `insert_slice`

Adding `TilingInterface` to `tensor.insert_slice` flips ~30 `dyn_cast`/`isa`
sites from "skip" to "match." Grouped by danger:

**A. IREE distribution/fusion paths (the real-bug pipeline) — highest danger:**
- `compiler/src/iree/compiler/Codegen/Common/GPU/GPUGreedilyDistributeToThreads.cpp:139`
  — `if (auto tilableOp = dyn_cast<TilingInterface>(op))` inside an IR **walk**;
  would attempt to tile any `insert_slice` it meets. This is in the GPU
  distribution path the `m[0::2,0::2]=True` dispatch traverses.
- `compiler/src/iree/compiler/Codegen/Common/TileAndFuseUtils.cpp:40,78,90,263,271,394`
  — producer/consumer fusion worklists keyed on `TilingInterface`; `insert_slice`
  would be treated as a fusable producer/consumer.
- `GPUFuseAndHoistParallelLoops.cpp:177,273`, `GPUTensorTile.cpp:150`,
  `GPUConvertToCoalescedDMA.cpp:470,1278`, `GPUFuseSubgroupConsumers.cpp:45`,
  `GPUTile.cpp:82,91,98,225,290`, `GPUTileAndConvertConvToMatmul.cpp:132`,
  `CPUPrepareUkernels.cpp:33,73,105`, `DecomposePackUnPackOps.cpp:170,205`,
  `MaterializeEncoding.cpp:206`.

**B. MLIR transform / fusion infrastructure:**
- `LinalgTransformOps.cpp:678,996,1057,1121,1648,2448,2786,2811,3344,3621,3899`
  — every `tile_*`/`fuse_*` transform op's `TilingInterface` gate now admits
  `insert_slice` as a legal target.
- `TileUsingInterface.cpp:1509,2027,2262,2290,2524` — producer/consumer fusion
  casts. **Notably `:2027`** requires `isa<DestinationStyleOpInterface>` too;
  `insert_slice` *is* `DestinationStyleOpInterface` (`TensorOps.td:843`), so it
  passes both gates and becomes a fusion-eligible consumer.
- `SwapExtractSliceWithProducerPatterns.cpp:27,78` — `replaceExtractSlice…` /
  `replaceInsertSlices…` would route an `extract_slice` of an `insert_slice`
  result into the new impl's `generateResultTileValue`. A1's impl **does not
  implement** `generateResultTileValue` (PadOp does, `:79-83`), so the default
  returns `failure()` — likely safe, but now newly attempted.

**Containment options for A2 (not in A1):**
1. Gate the impl's *use* (not its *registration*) — make
   `getIterationDomain`/`getTiledImplementation` return `failure()` unless a
   marker/attribute is present, so only the explicit transform anchor triggers
   it and greedy distributors no-op. (Trade-off: smells like a test-only op; A1
   rejected mechanism (d) for that reason — but a guarded production impl is
   different from a stand-in op.)
2. Add `insert_slice` to IREE's `isComputeOp`/anchor filters as an explicit
   allow-list rather than relying on the bare `TilingInterface` cast
   (`TileDispatchUsingForall.cpp:67-76`). A1 lists this as OUT (§9) but it is the
   actual blast-radius firewall and belongs in the rework's scope notes.

---

## 5. Other findings

- **Tensor-dialect guards G1/G2** (`SwapExtractSliceWithProducerPatterns.cpp:31-33`,
  `:99-101`, comment *"TilingInterface currently only supports strides being 1"*):
  A1 is **correct** these are off the initial-tiling path — they live in
  `replaceExtractSliceWithTiledProducer` (`:25`) / `replaceInsertSlicesWithTiledConsumer`
  (`:62`), called by the swap/fusion patterns, not by `tileUsingSCF`. The probe
  reaches `:616` via `generateLoopNestUsingForallOp`, which never calls these.
  Confirmed irrelevant to G3/G4. The "lockstep flip" (Stage 5.4) is cosmetic
  consistency, not a gate dependency.
- **Consumer-fusion rejection** `:2313-2317` (`"containingOp's result yield with
  stride"`) and consumer dest-extract `:2366-2367` (hardcoded unit stride):
  confirmed off the probe path (they are in `tileAndFuseConsumerOfSlicesImpl`).
  Their very existence re-confirms the refuted plan's FLAW 3 — consumer fusion
  explicitly rejects strided insert.
- **numThreads math (A1 §7):** CONFIRMED. `getTileOffsetAndSizesWithForAllOp`
  (`:491`) computes `offset = loopRange.offset + iv*givenTileSize` with
  `givenTileSize = ceilDiv(range, numThreads)` baked in by the caller; so
  `getResultTilePosition` receives effective-size-adjusted offsets. A1's claim
  that `iv*T*S` is correct "by construction" holds. The Stage-5.2 test discipline
  (force `ceilDiv(range,nt)≠1`) is sound.
- **Precedent for a strided result position:** NONE. `PadOp` is identity
  (`TensorTilingInterfaceImpl.cpp:63-64` copies offsets/sizes verbatim); linalg
  is projected-permutation only (`getMappedOffsetAndSize` enforces `AffineDimExpr`
  at `:169-171`). A1 is inventing new `TilingInterface` semantics (strided +
  iter/result rank-mismatched result tiles). Acceptable for a Phase-0 probe
  *because* the only intended producer of the new semantics is the single
  `:616` site A1 threads, and the default (unit) leaves all other implementors
  unchanged — but see F3: the *consumers* of the interface are the uncontained
  risk.

---

## 6. Specific questions / required changes for A2

1. **[BLOCKER] Fix the size inversion.** Replace, in the `insert_slice`
   `getResultTilePosition` impl, `resultSizes[d] = iterSizes[d]*stride[d]` with
   `resultSizes[d] = iterSizes[d]`. Cite `verifyInsertSliceOp`
   (`TensorOps.cpp:2885-2896`: `sizes` ⇒ source shape via
   `inferResultType(dstType, sizes)`) as the authority. Re-derive §3's expected
   writeback to `parallel_insert_slice … [iv*2,0][1,4][2,1]` and pin THAT in the
   R2 FileCheck (the current `[2,4]` FileCheck would match invalid IR or nothing).
2. **[BLOCKER] Fix the Case-1 IR.** `tensor.insert_slice … [0,0][2,4][2,1] :
   tensor<2x4xi32> into tensor<4x4xi32>` (sizes = source shape). Re-verify the
   `:2885`/`:2910` checks by hand. Update §3's offset/size table and the EXEC
   expected matrix (unchanged: rows {0,2}=1).
3. **[REQUIRED] Add a blast-radius containment section.** List the IREE greedy
   distributor (`GPUGreedilyDistributeToThreads.cpp:139`) and the
   `TileAndFuseUtils` worklists as sites that newly match `insert_slice`, and
   state the firewall: either a marker-gated impl (return `failure()` unmarked)
   or an explicit IREE anchor allow-list. This is on the GO-authorized path, not
   a footnote.
4. **[REQUIRED] Trace the EXEC lowering, don't just gate it.** Replace the §8
   "known risk" paragraph with a concrete trace: `parallel_insert_slice` (in
   `scf.forall`) → `ForallOpInterface::bufferize`
   (`BufferizableOpInterfaceImpl.cpp:1243-1296`, does NOT inspect strides) →
   strided `memref.subview`+copy → `-convert-*` → LLVM. Identify the *actual*
   first pass that could drop the stride (the cited `:672-674` gate is on
   standalone `InsertSliceOp`, off this path). Until traced, G4 is a black box.
5. **[REQUIRED] State the rank-reduction rule.** For rank-reduced
   `insert_slice`, give the explicit `getResultTilePosition` expansion over
   `getDroppedDims` (`TensorOps.cpp:3217-3219`) so the impl is not hand-waved as
   "modeled on PadOp" (PadOp has no iter/result rank mismatch). At minimum,
   restrict the G3/G4 probe to non-rank-reduced and say so in the gate.
6. **[NICE] Re-frame the no-overlap discussion.** Move it from §11 risk #1
   (speculative) to §3 (resolved): under `resultSizes = iterSizes`, tiles are
   point-set disjoint for all stride≥1, tile≥1 (gap = `T(S−1) ≥ 0`). The only
   contract-violating variant was the discarded `*stride` span.
7. **[NICE] Acknowledge the `:694-698` hostility comment** in §8: the standalone
   `InsertSliceOp` bufferization author flags tiled insert-slices as
   "catastrophically bad scheduling." It does not apply to the in-place
   `parallel_insert_slice` form, but it signals where reviewer pushback will
   come on upstreaming (A1 §11, last bullet).

---

### Summary table

| # | Finding | Severity | Blocks G3/G4? | Fix |
|---|---|---|---|---|
| F1 | `resultSizes=iterSizes*stride` → invalid writeback (`:2885`) | CRITICAL | Yes (G3) | `resultSizes=iterSizes` |
| F2 | Case-1 IR `sizes[4,4]` invalid (`:2885`,`:2910`) | CRITICAL | Yes (G3) | `sizes[2,4]` |
| F3 | Blast radius of `TilingInterface` on `insert_slice` | HIGH | No (integration) | marker-gate / allow-list |
| F4 | EXEC lowering untraced; gate mis-cited | MED/HIGH | Maybe (G4) | trace actual subview→LLVM path |
| F5 | rank-reduction + dynamic sizes unhandled | LOW (deferred) | No | state drop-dims rule |

The call chain is sound, the stride-source thesis is correct, and the fix to F1/F2
is mechanical. **Conditional GO pending A2 items 1–4.**
