# Reasoning Companion — A2 Rework Changes (per-finding changelog)

> **Purpose:** durable per-finding record of every change A2 made to
> `approach1_contract_phase1_plan.md` (v1 → v2). For each finding: the **OLD**
> text (A1 v1), the **NEW** text (A2 v2, as it now reads in the plan), and the
> `file:line` evidence that justifies it. The *decision rationale* lives in the
> plan's §12; this file is the diff/evidence trail.
>
> **Grounding rule:** every `file:line` below was opened personally in the
> `~/Developer/iree` / `third_party/llvm-project` tree this session. `[INFERENCE]`
> marks reasoning-only claims. No code changed, no builds run.

---

## Summary

| # | Finding | Kind | Severity |
|---|---|---|---|
| F1 | `resultSizes = iterSizes*stride` → `iterSizes` | **FIX (BLOCKER)** | CRITICAL |
| F2 | Case-1 anchor `sizes [4,4]` → `[2,4]` | **FIX (BLOCKER)** | CRITICAL |
| A/B | tile-size-1 primary → full-source genuine-scatter primary | **RESTRUCTURE (BLOCKER)** | CRITICAL |
| 7 | per-tile dest `%dest` → region iter arg `%o0`; validity = G4 criterion | **FIX** | HIGH |
| F3 | blast-radius containment — new §4a | **NEW SECTION** | HIGH |
| F4 | EXEC lowering trace — corrected `:672-674` mis-citation | **FIX + REWORDING** | HIGH |
| F5 | rank-reduction `getDroppedDims` rule stated; probe restricted | **NEW** | MED |
| 6a | numThreads EXEC promoted to core G4b | **RESTRUCTURE (JUSTIFIED)** | HIGH |
| 6b | §10 GO reworded — IR-level capability only | **REWORDING** | HIGH |

**Fixes vs rewordings:** F1, F2, 7, F4 are correctness *fixes* (v1 was wrong).
F3, F5 are *additions* (v1 was silent). 6a is a *promotion* (v1 deferred). 6b
and A/B are *restructurings/rewordings* (v1's claims were over-strong or
mis-framed). **Least-sure item:** F4 step 3 (memref→LLVM survival of the strided
subview) — confirmed only by reading that bufferization *forwards* the stride,
not by running the lowering; marked `[INFERENCE]`. **numThreads-G4 promotion
(6a): JUSTIFIED** — see entry.

---

## F1 — `resultSizes = iterSizes` (BLOCKER)

**OLD (v1, §1.3 / §3 / Stage 3.1):**
> `resultSizes[d] = iterSizes[d] * insert.getMixedStrides()[d]` (the "span"; §3
> justifies) … `resultSizes = [offsets-derived span] = [2, 4]` … per-tile
> writeback `parallel_insert_slice %tile<1x4> into %dest[iv*2,0] [2,4] [2,1]`.

**NEW (v2):**
> `resultSizes[d] = iterSizes[d]` (the SOURCE tile shape; NOT `iterSizes*stride` —
> the dest span `(size-1)*stride+1` is implicit and checked in-bounds) …
> `resultSizes = iterSizes (source tile shape) = [1, 4]` (sub-case B) /
> `[2, 4]` (sub-case A, = source shape).

**Evidence:** `lib/Dialect/Tensor/IR/TensorOps.cpp:2885-2896` —
`verifyInsertSliceOp` computes the expected source type as
`ExtractSliceOp::inferResultType(dstType, staticSizes)` (**sizes only, no
strides**) and checks `isRankReducedType(expected, srcType)`. So the
source/`%tile` shape MUST equal `sizes`; the dest span
`offset+(size-1)*stride` is implicit (in-bounds check at `:2910-2912`).
`insert_slice` is the *inverse* of `extract_slice`; A1 had applied *extract*
semantics (`sizes`=dest span) to an *insert* op. Given as orchestrator ground
truth and re-confirmed by opening `:2885-2896`.

**Knock-on:** no-overlap contract (`TilingInterface.td:146-147`) is satisfied
under the corrected convention for all stride ≥ 1, tile ≥ 1 (point-set gap =
`str ≥ 1`); the only contract-violating variant was the discarded `*stride`
span. (§3 "No-overlap contract — RESOLVED", §11 risk #1 → RESOLVED.)

---

## F2 — Case-1 anchor `sizes [2,4]` (BLOCKER)

**OLD (v1, §3 input IR):**
> `%r = tensor.insert_slice %filled into %dest[0, 0] [4, 4] [2, 1] :
> tensor<2x4xi32> into tensor<4x4xi32>` … "span 4/stride2 = 2 source rows".

**NEW (v2):**
> `%r = tensor.insert_slice %filled into %dest[0, 0] [2, 4] [2, 1] :
> tensor<2x4xi32> into tensor<4x4xi32>` (sizes = source shape). The comment
> "span 4/stride2" (the inverted mental model) is removed.

**Evidence:** same authority (`TensorOps.cpp:2885-2896`). v1's `[4,4]` failed
both checks: `inferResultType(dest<4x4>, [4,4])` = `<4x4>` ≠ source `<2x4>`; and
in-bounds `(4-1)*2+1 = 7 > 4`. Corrected `[2,4]`:
`inferResultType(dest<4x4>, [2,4])` = `<2x4>` ✓; in-bounds `(2-1)*2+1 = 3 ≤ 4` ✓.
Hand-derived in §3 (sub-case A) and re-verified against `:2885`/`:2910`.

---

## A/B reframe — genuine scatter is the only capability proof (BLOCKER)

> Added in response to an orchestrator directive (Main) received mid-pass;
> verified against the verifier before incorporating. This is the most
> load-bearing restructure.

**OLD (v1):** the tile-size-1 sub-case (`tile_sizes [1,4]`, per-tile writeback
`[1,4][2,1]`, 2 tiles) was the *primary* G3/G4 probe, asserting `[2,1]` as the
stride proof.

**NEW (v2):**
- **Sub-case A (PRIMARY):** full-source 1-tile genuine scatter —
  `tile_sizes [2,4]`, writeback `parallel_insert_slice %tile<2x4> into
  %o0[0,0][2,4][2,1]`. Tile row 1 → dest row 2; a hardcoded `[1,1]` would put it
  at dest row 1 (WRONG). The stride is load-bearing.
- **Sub-case B (SECONDARY, demoted):** tile-size-1, `tile_sizes [1,4]`, writeback
  `[1,4][2,1]`. Stride is **vestigial**; B proves only the strided *offset*
  (`iv*2`), NOT the capability.

**Evidence (the vestigial-stride math):** `OffsetSizeAndStrideOpInterface`
placement is `offset + j*stride` per element. `verifyInsertSliceOp`
(`TensorOps.cpp:2885-2896`) checks `sizes` against the source (via
`inferResultType`, sizes-only) and in-bounds (`:2910`); it does NOT check stride
*meaningfulness*. On a size-1 strided dim, `j ∈ {0}` so placement = `offset`,
identical to unit stride — a hardcoded `[1,1]` writeback passes B byte-for-byte.
Only a strided dim with size ≥ 2 (sub-case A) makes the stride channel
load-bearing. Confirmed by reading `:2885-2896` and `:3960-3982`.

**Where:** §3 (full rewrite), §5 Stage 1.1 (two transforms A+B) + G3 gate
(sub-case A primary), §10 GO ("load-bearing proven ONLY via sub-case A"; new
NO-GO mode "G3 holds but only via B → FALSE GREEN").

---

## Item 7 — per-tile dest = region iter arg; validity = G4 criterion

**OLD (v1, §3 writeback / §1.3 box):**
> `tensor.parallel_insert_slice %tile into %dest[...]` (dest labeled as the outer
> `%dest`). Validity of a strided `parallel_insert_slice` inside `scf.forall` was
> a passing remark, not a gate.

**NEW (v2):**
> `tensor.parallel_insert_slice %tile into %o0[...]` where `%o0` is the forall's
> **region iter arg** (`shared_out`), not the outer `%dest`. Validity (verifier +
> bufferization survival) is an **explicit G4 acceptance criterion**.

**Evidence:** `lib/Dialect/SCF/Transforms/TileUsingInterface.cpp:597` —
`innerDestinationTensors = forallOp.getRegionOutArgs()` (region block args, one
per `shared_out`); the writeback zip at `:613-621` iterates
`innerDestinationTensors`, so `ParallelInsertSliceOp::create` (`:619-621`)
receives the region iter arg as `dest`, not the outer `%dest` (which becomes the
forall `shared_outs` at `:585-592`). Validity: verifies at
`TensorOps.cpp:3960-3982`; bufferizes at
`BufferizableOpInterfaceImpl.cpp:969-1027` (see F4). End-to-end survival is
proven only by a green G4 (§8, §10).

---

## F3 — blast-radius containment (NEW section §4a)

**OLD (v1):** absent. A1's R1 invariant guarded the *channel threading*, not the
global effect of giving `insert_slice` `TilingInterface`.

**NEW (v2, §4a):** a containment section specifying two firewalls:
1. **Marker-gated impl (PRIMARY):** `getIterationDomain`/`getTiledImplementation`/
   `getResultTilePosition` return `failure()` unless a discardable marker attr
   (set by the transform anchor) is present. Greedy distributors/fusion worklists
   hit `failure()` → no-op; only the explicit probe exercises the strided path.
   The impl also *omits* the operand-tile/fusion methods (so fusion consumers
   get the default `failure()`).
2. **IREE anchor allow-list (DEFENSE-IN-DEPTH):** `isComputeOp`/
   `TileDispatchUsingForall.cpp:67-76` and the `TileAndFuseUtils` worklists
   filter on an explicit allow-list, not the bare `TilingInterface` cast.

**Evidence (sites verified this session):**
- `compiler/src/iree/compiler/Codegen/Common/GPU/GPUGreedilyDistributeToThreads.cpp:114-154`
  — `processRegion` walk; `dyn_cast<TilingInterface>(op)` at `:139`, routed to
  `tileToThreads` at `:145`. In the real bug's GPU distribution path.
- `compiler/src/iree/compiler/Codegen/Common/TileAndFuseUtils.cpp:40,78,90,271,394`
  — fusion worklist casts.
- `lib/Dialect/SCF/Transforms/TileUsingInterface.cpp:2027` — consumer-fusion gate
  ALSO requiring `isa<DestinationStyleOpInterface>`; `insert_slice` IS
  DestinationStyle (`TensorOps.td:843`), so it passes both.
- PadOp precedent for the omitted-methods argument:
  `lib/Dialect/Tensor/IR/TensorTilingInterfaceImpl.cpp:79-83`
  (`generateResultTileValue`) — the insert_slice impl deliberately does NOT
  implement it.

---

## F4 — real EXEC lowering trace (corrects the `:672-674` mis-citation)

**OLD (v1, §8 "Known EXEC risk"):**
> "one-shot bufferization has an `allStridesOne` gate on `insert_slice` in-place
> aliasing (`Tensor/Transforms/BufferizableOpInterfaceImpl.cpp:672-674`) … if it
> bails, bufferization falls back out-of-place."

**NEW (v2, §8 "EXEC lowering path — TRACED"):** the G3 writeback is a
`parallel_insert_slice` *inside* `scf.forall`, which does **not** go through the
standalone `InsertSliceOp` path A1 cited. The traced path:
1. `ForallOpInterface::bufferize` (`SCF BufferizableOpInterfaceImpl.cpp:1243-1296`)
   — ignores strides (`:1209-1212`: terminators are "analysis only").
2. `ParallelInsertSliceOpInterface::bufferize`
   (`Tensor BufferizableOpInterfaceImpl.cpp:969-1027`) — forwards
   `getMixedStrides()` verbatim into `memref.subview` (`:998-1002`) + `memref.copy`
   (`:1005-1006`). **No stride gate.**
3. memref → LLVM (`-convert-memref-to-llvm` / `-expand-strided-metadata`) — the
   first place a stride could be dropped/rejected `[INFERENCE]`; the real G4 risk.

**Evidence (all opened this session):**
- `SCF/Transforms/BufferizableOpInterfaceImpl.cpp:1243-1296` (`ForallOpInterface::bufferize`:
  `to_tensor(memref)` + `mergeBlocks` at `:1260-1290`, no stride inspection;
  `:1209-1212` comment).
- `Tensor/Transforms/BufferizableOpInterfaceImpl.cpp:949-1037`
  (`ParallelInsertSliceOpInterface::bufferize`: strides → `SubViewOp` `:998-1002`
  + `createMemCpy` `:1005-1006`; registered at `:1209-1210`).
- `:655-733` (`InsertSliceOpInterface` — the `allStridesOne` analysis gate
  `:672-674` and `insertSliceOpRequiresRead` `:656-675` live HERE, on the
  **standalone** `InsertSliceOp`, off the writeback path; `:694-698` "catastrophic
  scheduling" hostility comment).

**G4 risk, stated honestly:** bufferization preserves the stride by construction
(no gate). The risk is entirely in memref→LLVM (step 3), unverified until G4
green. A COMPILE-fail is a distinct NO-GO mode that still proves G3.

---

## F5 — rank-reduction rule (NEW; probe restricted)

**OLD (v1):** rank-reduction acknowledged as a stretch ("deferred, A1
acknowledges"); the impl hand-waved as "modeled on PadOp." No rule stated.

**NEW (v2, §11 risk #3):** the `getDroppedDims` rule is stated, and the G3/G4
probe is **restricted to non-rank-reduced**. Rule: for **preserved** dims
`rOff[d]=base[d]+iter_off[d]*str[d]`, `rSize[d]=iter_size[d]`; for **dropped**
dims `rOff=base[d]`, `rSize=1`.

**Evidence:** `lib/Dialect/Tensor/IR/TensorOps.cpp:3217-3219`
(`InsertSliceOp::getDroppedDims` → `::getDroppedDims(getSourceType().getShape(),
getMixedSizes())`) and `:142-180` (the helper: a size-dim is dropped iff static-1
and not a preserved source dim). This creates a **rank-mismatch** — iteration
domain = source rank, result position = dest rank — that PadOp
(`TensorTilingInterfaceImpl.cpp:24-84`, iter domain = result rank, identity
`:33-44`) never faces. Inventing strided AND rank-mismatched result-tile
semantics in one probe is unjustified → defer with the rule on record.

---

## 6a — numThreads EXEC promoted to core G4b (JUSTIFIED)

**OLD (v1, §7 / Stage 5.2):** numThreads was a **Stage-5 stretch** ("drive
Case-1 with `numThreads=[2]` … assert the offset is correct"). G4 was a single
`tile_sizes` EXEC.

**NEW (v2, §7 / §5 Stage 4.2):** numThreads EXEC is **promoted to core G4b** as a
**genuine-scatter multi-tile** case: `source<4x4> → dest<8x4>`, stride `[2,1]`,
`num_threads [2]` → `ceilDiv(4,2)=2` per-tile source size, 2 tiles, EACH a
genuine within-tile scatter (size-2 on the strided dim), offset `iv*4`. Asserts
dest rows `{0,2,4,6}` = 1.

**Verdict: JUSTIFIED (not deferred).** Reasons:
1. **Different code path.** The `numThreads` branch computes per-tile offsets via
   *different code* than `tile_sizes` — `getTileOffsetAndSizesWithForAllOp`
   `:473-544`, `useNumThreads` at `:574`, `offsetExpr = d0+d1*s0` at `:491`, plus
   residual/boundary `min/max` handling at `:514-538`. A `tile_sizes`-only green
   does NOT exercise this path; it leaves the real-IREE offset composition
   unproven.
2. **Cheap.** One extra lit + EXEC on the same harness (§8). Elevates §7's
   `iv*T*S` "by construction" math from `[INFERENCE]` to evidence.
3. **Avoids a false green on the promoted path.** The case is a genuine scatter
   (per-tile size ≥ 2 on the strided dim), so G4b is not itself a vestigial-stride
   probe (cf. the A/B reframe).

**What is NOT covered (stays OUT):** the dynamic-size half (`tensor<?x?xi8>`,
runtime `ceildivi`) — separate unproven boundary (§9, §10, 6b).

---

## 6b — §10 GO reworded (IR-level capability only)

**OLD (v1, §10):**
> "GO iff BOTH hold: G3 (IR) … G4 (EXEC) writes rows {0,2}=1. On GO: Phase 0
> answers YES — the contract change is sufficient for the tiler to emit a correct
> strided writeback for a real slice op."

**NEW (v2, §10):** GO proves the **IR-level + static-EXEC capability** on BOTH
offset paths (`tile_sizes` G4a + `numThreads` G4b), with the stride
load-bearing (proven ONLY via sub-case A). GO does **NOT** prove
`m[0::2,0::2]=True` compiles: dynamic `tensor<?x?xi8>` (runtime `ceildivi`),
IREE anchor selection + `computeOp` filter widening, and end-to-end lowering
survival are a **separate, unproven boundary**, not folded into GO.

**Evidence:** the static transform-interpreter probe exercises neither dynamic
iteration domains nor IREE's anchor selection
(`TileDispatchUsingForall.cpp:67-76`; `isComputeOp` at `Utils.cpp:980-982`).
Promoting those into the GO claim would be a false-green; stating them as a
boundary is the honest scope line.

---

## Note on mechanism (c) — NOT re-litigated

Per the assignment, the chosen mechanism (insert_slice as a `TilingInterface`
anchor tiled via INITIAL tiling, stride from its own `strides` attribute) is
settled. The verified call chain (`tile_using_forall` → `tileToForallOpImpl`
→ `dyn_cast<TilingInterface>` → `tileUsingSCF` → `generateLoopNestUsingForallOp`
→ `:616-617` writeback) is unchanged and confirmed by R1 §2. A2 addressed the
*size semantics, the probe shape, the lowering trace, the blast radius, and the
scope claims — not the mechanism itself.
