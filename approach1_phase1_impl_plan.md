> ⛔ **SUPERSEDED 2026-06-29 — DO NOT EXECUTE.** An expert review (verified by
> re-reads) refuted this plan's central thesis (SCF-only consumer fusion can
> emit a strided writeback without the contract change). Three fatal flaws:
> (1) `tensor.insert_slice` is NOT `TilingInterface`
> (`TensorTilingInterfaceImpl.cpp:314` registers it only for `PadOp`), so it
> can't be the fused consumer; (2) `:2311` reads the loop-**internal**
> unit-stride `parallel_insert_slice` candidate (`getProducingParallelInsertSlice`
> `:2487`), never the external `[2,1]` store — consumer fusion preserves but
> cannot create strides; (3) `fuse_into_containing_op` does producer, not
> consumer, fusion. **The `TilingInterface.td` contract change is a prerequisite,
> not a fallback.** Full review: `expert_review_phase1_plan.md`. What survives:
> R2 (complete per-tile IR assertion) + the §3 hand-derived offset math
> (`iv*stride`) — reuse as the acceptance contract for the replacement approach.
>
---
# Approach 1 — Phase 1 Implementation Plan (DRAFT v3, for iteration)

> **Status:** DRAFT v3. Not for execution; iterate before any code changes.
> Every file:line below was re-opened and confirmed in the
> `~/Developer/iree` / `third_party/llvm-project` tree during planning.
>
> **Source of truth:** `approach1_tensor_level_fusion.md` (feasibility
> investigation). This plan implements its **Phase 0 de-risk** only.
>
> **v1→v2→v3 changes:**
> - **v2:** the stride source is already in SCF (`TileUsingInterface.cpp:2308-2311`,
>   discarded `:2313-2320`); linalg maps are projected permutations
>   (`TilingInterfaceImpl.cpp:268-275`) so `getResultTilePosition` is unit for all
>   linalg → the `TilingInterface.td` contract change is the **go/no-go fallback
>   (§6), not a stage**. SCF-only probe front-loaded.
> - **v3 (this):** corrected site reachability. Initial tiling
>   (`generateLoopNestUsingForallOp:556-624`) emits the writeback only and leaves
>   dest handling to the implementor — **no SCF dest-extract**, unit strides for
>   linalg. The dest-extract sites (`:2366`, `:1565`) are **fusion-only**, behind
>   the rejection gates. The ½× read case is therefore NOT independently testable
>   via initial tiling (v2's §3.2 was a false-RED no-op). **GREEN-A is now a single
>   consumer-fusion stage** covering both the 2× write (`:1006`) and the strided
>   dest-extract (`:2366`), with two complementary tests.

**Goal:** Prove, red/green TDD, that upstream MLIR's SCF tiler can *emit* a
non-unit-strided tensor writeback for two static cases — **2× dilation
(write)** and **½× dilation (strided-destination read)** — by threading the
candidate's own strides through the **consumer-fusion** path, **without touching
`TilingInterface.td`**. The open question is whether SCF's local offset/size
composition is correct — resolved by the hand-derived IR + `mlir-cpu-runner` gate.

**Architecture (SCF-only, consumer-fusion path):** the stride lives in the
candidate `insert_slice`/`parallel_insert_slice` attributes. SCF reads it
(`:2311`), checks-then-discards it (`:2313-2320`), forwards the raw destination
span (un-stride-divided) to `getIterationDomainTileFromOperandTiles` (`:2330`)
and `getResultTilePosition` (`:2347`), and hardcodes `1` at the writeback
(`:1006-1007`) and dest-extract (`:2366-2367`) sites. The fix: carry strides
through SCF's `YieldTiledValuesFn`, stride-divide the span before the two
interface calls, post-multiply the strided-dim offset, emit the stride at the
two sites, flip `:2313` last.

**Scope IN:** vendored `third_party/llvm-project` MLIR, SCF
`TileUsingInterface.cpp` **consumer-fusion path only**, static stride `2`. No
`TilingInterface.td` edit unless §6 fallback triggers.

**Scope OUT (§9):** the `TilingInterface.td` contract change (fallback only);
initial-tiling sites `:616`/`:447` (different typedef `GenerateTiledBodyFn`,
unreached by consumer fusion — deferred); producer-fusion sites `:951`/`:1565`
(wired in A.2 for typedef-consistency, but the `:1502` rejection is NOT flipped
and they are untested); IREE filter widening; source-load co-distribution;
arbitrary/coprime strides; #51660.

---

## 0. Inviolable rules (gate EVERY step)

The doc and review rounds converged on one failure mode: **dropping a stride
silently writes to the wrong memory — strictly worse than today's hard compile
error.**

- **R1 — Propagate-first, flip-rejection-last.** `TileUsingInterface.cpp:2313-2317`
  (consumer) and `:1502-1504` (producer) are *safe hard failures today*.
  Candidate strides are read (`:2311`), checked (`:2314`), **discarded** — only
  offsets/sizes enter `allOffsets`/`allSizes` (`:2319-2320`); the dest extract at
  `:2366-2367` hardcodes `1`. **No commit may exist where a gate is removed but
  propagation is not wired through ALL fusion-path sites** — consumer writeback
  `:1006` + dest-extract `:2366` (reached by BOTH §3.1 and §3.2, since
  `tensor.insert_slice` is DPS — `TensorOps.td:843`) AND their producer twins
  `:951`/`:1565` (same `YieldTiledValuesFn` typedef) — plus the `:2308-2336`
  offset/size composition. The initial-tiling pair `:616`/`:447` uses a
  *different* typedef (`GenerateTiledBodyFn`) and is unreached by consumer fusion
  (linalg initial tiling is unit stride) — deferred (§9), not a confound. Flip
  `:2313` only as the final atomic action of §4, gated on the execution test.

- **R2 — GREEN asserts the COMPLETE writeback IR, hand-derived per tile.** A
  FileCheck matching only `[2,1]` is a false-green: for tile-size-1 the stride
  manifests as an **offset** (`iv*2`), and `getResultTilePosition` (`:2347`) +
  `getIterationDomainTileFromOperandTiles` (`:2330`) receive the candidate's
  destination span **without stride-division** (`allSizes` at `:2331` = raw
  `getMixedSizes()`). The offset×stride composition is the exact place a silent
  miscompile hides. **Every GREEN test asserts offsets AND sizes AND strides for
  a small static case, derived by hand.**

- **R3 — Each GREEN culminates in a `mlir-cpu-runner` execution test.** IR-shape
  proves "SCF emits strided IR"; only execution proves "it writes the right
  cells." **EXEC is also what decides §6: if it passes, the SCF-only path is
  proven and the contract change is skipped.**

- **R4 — One regression invariant, always green.** A unit-stride twin (`[1,1]`)
  of every strided test must stay green at every commit.

---

## 1. Inventory (re-verified this session)

**Site reachability — the v3 correction:**

| Path | Writeback site | Dest-extract site | Gate | In scope? |
|---|---|---|---|---|
| Initial tiling (`generateLoopNestUsingForallOp:556-624`, `generateLoopNestUsingForOp:388`) | `:616-617` / `:447-448` | **none** (dest = implementor's `getTiledImplementation`) | — | NO (linalg → unit stride; not exercised) |
| **Consumer fusion** (`tileAndFuseConsumerOfSlicesImpl:2205-2416`) | **`:1006-1007`** | **`:2366-2367`** | **`:2313-2317`** | **YES (this plan)** |
| Producer fusion (`yieldReplacementForFusedProducer`) | `:951-955` | `:1565-1566` | `:1502-1504` | NO (producer rejection; §9) |

> Verified: `generateLoopNestUsingForallOp` (`:556-624`) takes a
> `GenerateTiledBodyFn` (`:563`), calls it (`:607`), emits only the writeback
> (`:616-621`) — **no dest-extract**. Consumer fusion's `newYieldValuesFn`
> (`:2298-2388`) contains the dest-extract (`:2363-2374`) AND prepares the
> writeback inputs (`:2381-2386`) via `addInitOperandsToLoopNest` (`:2390`) →
> `yieldTiledValuesAndReplaceLoop<scf::ForallOp>` (`:1006`).

**The stride source — already in SCF, no contract needed:**
- `:2308-2320`: reads `getMixedOffsets/Sizes/Strides()` (`:2309-2311`), checks
  all strides `1` (`:2314`), discards strides, forwards offsets/sizes to
  `allOffsets`/`allSizes` (`:2319-2320`).
- `:2330-2332`: `getIterationDomainTileFromOperandTiles(... allOffsets,
  allSizes ...)` — `allSizes` is the **raw destination span, NOT stride-divided**.
  For a `[0,0][4,4][2,1]` candidate this hands `[4,4]` where the producer tile is
  `[2,4]` (span/stride). **Central SCF-local fix point.**
- `:2347-2349`: `getResultTilePosition` — unit-stride offsets (fed by the above);
  SCF must post-multiply the strided-dim offset by the candidate stride.

**SCF-internal stride channel:** `YieldTiledValuesFn` typedef `:336-340` — extend
with a `resultStrides` field. This is a **SCF-internal** typedef, NOT the
`TilingInterface.td` contract (SCF-internal = low blast radius = this plan;
contract = high blast radius = §6 fallback only). The typedef is shared by all
fusion-path emission sites, so A.2 wires **all four** to the channel — consumer
writeback `:1006` + dest-extract `:2366`, producer writeback `:951` +
dest-extract `:1565` (the producer pair carries unit strides; it is
`:1502`-gated and untested, but wiring it makes R1 literal and removes any
EXEC-A confound).

**Why both tests reach both consumer sites:** `tensor.insert_slice` implements
`DestinationStyleOpInterface` (`TensorOps.td:843`), so when it is the tiled
consumer the DPS guard at `TileUsingInterface.cpp:2358-2359` succeeds and the
dest-extract at `:2366-2367` fires alongside the writeback at `:1006-1007`.
**Both §3.1 and §3.2 reach BOTH `:1006` and `:2366`** (verified); they differ
only in which dimension is strided, not in which sites they exercise.

**Why no `Test_TilingNoDpsOp` extension (v1 had one):** dropped. The stride
source is the real candidate `tensor.insert_slice`, not a synthetic test op.

**Harness (re-verified):** `.iree-build/llvm-project/bin/{FileCheck,llvm-lit}`
built; `mlir-opt` + `mlir-cpu-runner` are ninja targets, **not yet built**. MLIR
lit suite is **not configured** (no `lit.site.cfg.py`) → commands use direct
`mlir-opt | FileCheck` (config-independent) as canonical.

---

## 2. Stage 0 — Harness baseline (no behavior change)

**Files:** none (build only).

- [ ] **Step 0.1: Build the tools.**
  ```bash
  cmake --build /Users/alex/Developer/.iree-build --target mlir-opt FileCheck llvm-lit mlir-cpu-runner
  ```
  Expected: `.iree-build/llvm-project/bin/{mlir-opt,FileCheck,llvm-lit,mlir-cpu-runner}` exist.

- [ ] **Step 0.2: Smoke a known-good tiling test** (proves the runner before any change):
  ```bash
  .iree-build/llvm-project/bin/mlir-opt --transform-interpreter --split-input-file -canonicalize \
    third_party/llvm-project/mlir/test/Dialect/Linalg/transform-op-fuse.mlir \
    | .iree-build/llvm-project/bin/FileCheck \
      third_party/llvm-project/mlir/test/Dialect/Linalg/transform-op-fuse.mlir
  ```
  Expected: exit 0. If this fails, stop — fix harness first.

- [ ] **Step 0.3: Capture today's RED evidence.** Run §3.1's `@strided_write_2x`
  through consumer fusion *now*; record that `:2313-2317` rejects → transform
  no-ops (`insert_slice` stays outside the loop). This is the pre-change baseline.

---

## 3. The canonical static cases (hand-derived once; every GREEN asserts these)

> Both are **consumer-fusion** tests (fuse the `tensor.insert_slice` consumer
> into the producer's tiled loop) and, because `insert_slice` is DPS
> (`TensorOps.td:843`), **both reach the writeback `:1006` AND the dest-extract
> `:2366`** (§1) — they differ only in which dimension is strided. Initial tiling
> cannot exercise either — see §1.

### 3.1 — 2× dilation WRITE case (writeback `:1006` + dest-extract `:2366`)

**Input IR** (scatter-fill every other row of 4×4 with `1.0`):
```mlir
func.func @strided_write_2x(%dest: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %cst = arith.constant 1.0 : f32
  %init = tensor.empty() : tensor<2x4xf32>
  %filled = linalg.fill ins(%cst : f32) outs(%init : tensor<2x4xf32>) -> tensor<2x4xf32>
  // insert_slice semantics: filled(2,4) scatters into dest rows {0,2} (span 4 / stride 2 = 2)
  %r = tensor.insert_slice %filled into %dest[0, 0][4, 4][2, 1]
        : tensor<2x4xf32> into tensor<4x4xf32>
  return %r : tensor<4x4xf32>
}
```

**Expected IR after fusing the `insert_slice` consumer, two sub-cases** (both
asserted — they exercise disjoint parts of the offset×stride composition):

*Sub-case A — tile dim-0 by 2 (1 tile).* Trivial offset; proves the **writeback
site `:1006` emits non-unit stride**:
```mlir
// inside the fused scf.forall, 1 iteration:
tensor.parallel_insert_slice %tile into %dest[0, 0][4, 4][2, 1]
      : tensor<2x4xf32> into tensor<4x4xf32>
```

*Sub-case B — tile dim-0 by 1 (2 tiles).* **Miscompile-detection case.** The
1-row tile scatters across a 2-row span, so per-tile stride is still `[2,1]`
AND the dim-0 offset is `iv*2`:
```mlir
// iteration iv ∈ {0,1}:
//   CRITICAL: dim-0 offset is iv*2, NOT iv. Unit-stride assumption → iv → wrong row.
%off = arith.muli %iv, %c2 : index
tensor.parallel_insert_slice %tile into %dest[%off, 0][2, 4][2, 1]
      : tensor<1x4xf32> into tensor<4x4xf32>
```
> **R2 enforcement:** the sub-case-B FileCheck MUST assert BOTH the
> `arith.muli %iv, %c2` offset AND the `[2,1]` strides. Matching `[2,1]` while
> the offset is `iv` would PASS on paper and miscompile in silicon.
>
> **SCF-local origin of `%off`:** `getResultTilePosition` (`:2347`) returns the
> unit-stride offset `iv*tileSize`; SCF post-multiplies the strided dim by the
> candidate stride (`:2311`) → `iv*tileSize*stride`. tileSize=1, stride=2 → `iv*2`.

### 3.2 — ½× dilation READ case (dest-extract `:2366`) — reframed as consumer fusion

> v2 framed this as initial tiling of a fill — that was a **false-RED no-op**:
> the `[1,2]` lived in the outer `extract_slice`/`insert_slice` that tiling never
> touches, so the test would pass today. The dest-extract site `:2366` is only
> reached by **fusing the `insert_slice` consumer** so SCF re-creates the
> destination slice.

**Input IR** — a DPS fill whose result is inserted back into a strided subregion:
```mlir
func.func @strided_read_half(%dest: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %cst = arith.constant 1.0 : f32
  %sub = tensor.extract_slice %dest[0, 0][4, 4][1, 2] : tensor<4x4xf32> to tensor<4x2xf32>
  %r = linalg.fill ins(%cst : f32) outs(%sub : tensor<4x2xf32>) -> tensor<4x2xf32>
  %inserted = tensor.insert_slice %r into %dest[0, 0][4, 4][1, 2]
        : tensor<4x2xf32> into tensor<4x4xf32>
  return %inserted : tensor<4x4xf32>
}
```
**Expected after fusing the `insert_slice` consumer:** the tiled fill's DPS init
gets a dest `extract_slice` of `%dest` carrying `[1,2]` (site `:2366`), matching
the writeback stride on dim-1. Today `:2366` hardcodes `[1,1]` → reads the wrong
cells. Assert the dest-extract **stride `[1,2]`** AND an offset consistent with
the writeback.

### 3.3 — Expected execution result (R3 semantic gate)

- §3.1 over all-zero `%dest` → rows `{0,2}` filled: `1111 / 0000 / 1111 / 0000`.
- §3.2 → columns `{0,2}` filled, `{1,3}` preserved.

---

## 4. GREEN-A — SCF-only consumer-fusion probe (2× write + ½× read)  ← the Phase 0 de-risk

**Answers: "can SCF locally emit a correct strided writeback + strided
dest-extract via consumer fusion, without the contract change?"**

**Files:**
- Modify: `third_party/llvm-project/mlir/lib/Dialect/SCF/Transforms/TileUsingInterface.cpp`
  — `YieldTiledValuesFn` typedef `:336-340` (add `resultStrides`); **all four
  fusion-path sites** to the channel — writeback `:1006-1007` + `:951-955`,
  dest-extract `:2366-2367` + `:1565-1566`; `:2308-2336` (carry candidate
  strides; stride-divide `allSizes` before `:2330`; post-multiply strided-dim
  offset after `:2347`); flip rejection `:2313-2317` LAST.
- Create: `third_party/llvm-project/mlir/test/Dialect/SCF/transform-strided-consumer-fusion.mlir`.

- [ ] **Step A.1 (RED-A): Write the two failing tests.** §3.1 (2× write, both
  sub-cases) and §3.2 (½× read) driven through consumer fusion
  (`transform.structured.fuse_into_containing_op` on the `tensor.insert_slice`),
  asserting the §3.1/§3.2 hand-derived IR.
  Run:
  ```bash
  .iree-build/llvm-project/bin/mlir-opt --transform-interpreter <test>.mlir \
    | .iree-build/llvm-project/bin/FileCheck <test>.mlir
  ```
  Observe FAIL: today `:2313-2317` rejects → transform no-ops. Capture as RED.

- [ ] **Step A.2 (PROPAGATE — do NOT flip yet):**
  (a) Extend `YieldTiledValuesFn` (`:336-340`) with `resultStrides`.
  (b) Wire **all four fusion-path sites** to the channel — consumer writeback
  `:1006-1007` + dest-extract `:2366-2367`, and producer writeback `:951-955` +
  dest-extract `:1565-1566` (same typedef; producer pair carries unit strides,
  `:1502`-gated, untested — wired for R1-literal completeness).
  (c) At `:2308-2336`: keep candidate strides (`:2311`); **stride-divide
  `allSizes`** (ceil(span/stride) on strided dims) before `:2330` so
  `getIterationDomainTileFromOperandTiles` gets the producer-correct `[2,4]` not
  the destination span `[4,4]`; after `getResultTilePosition` (`:2347`),
  **post-multiply the strided-dim offset** by the candidate stride (yields
  `iv*tileSize*stride`, the §3.1-B `%off`); thread strides to all four sites.
  At this point `:2313` still gates the strided case → only the unit-stride path
  exercises this code → **R4 twin must stay green.**
  Commit: `git commit -m "mlir/SCF: thread candidate strides through consumer-fusion writeback + dest-extract (gate retained)"`.
  **R1 check: `:2313` NOT flipped — safe.**

- [ ] **Step A.3 (FLIP — atomic):** Change `:2313-2317` reject→propagate.
  Rebuild. RED-A must go green (all three sub-cases). **Immediately** run R4.

- [ ] **Step A.4 (EXEC-A — R3, decides §6):** `mlir-cpu-runner` on §3.1 (assert
  `1111/0000/1111/0000`) and §3.2 (assert columns `{0,2}` filled). Skeleton
  (capture exact `-shared-libs` from an existing `mlir/test/Target/` test):
  ```bash
  .iree-build/llvm-project/bin/mlir-cpu-runner ... %s -e entry -entry-point-result=void \
    -shared-libs=...mlir_c_runner_utils... | FileCheck %s --check-prefix=EXEC
  ```
  **Decision:** EXEC passes → §6 fallback SKIPPED, commit A.3. EXEC fails but
  IR-shape passes → silent miscompile in A.2(c); do NOT commit; go to §6.

- [ ] **Step A.5: Commit the flip** (only if EXEC-A green):
  `git commit -m "mlir/SCF: support strided consumer in tileAndFuseConsumer"`.

> **Go/no-go gate GA:** EXEC-A green = **YES, SCF can emit correct strided
> writeback + dest-extract locally.** This is the Phase 0 binary answer.

---

## 5. FALLBACK — the `TilingInterface.td` contract change (go/no-go, NOT a stage)

**Triggered ONLY if** GA EXEC fails in a way that proves SCF-local composition
is fundamentally insufficient (e.g., the stride-divide at A.2(c) cannot be made
consistent with `getIterationDomainTileFromOperandTiles`'s semantics for some op
class). **If GA EXEC passes, SKIP this section.**

If triggered: add `resultStrides` to `getResultTilePosition`
(`TilingInterface.td:118-162`), populate in implementors, thread through SCF.
This is the doc's highest-blast-radius item — justified only with evidence the
SCF-only path is inadequate. Re-derive §3 IR under the contract model; re-run EXEC.

> §4 makes §5 an *informed* decision: spend ~1 day proving the SCF-only path
> before committing to the multi-week contract battle. Don't front-load blast
> radius if the evidence (§1) says you might not need it.

---

## 6. Phase 0 verdict + capture

- [ ] **Step 6.1:** Record the binary answer (GA green = yes). Write into
  `approach1_tensor_level_fusion.md` §7.4, converting Phase-0 `[INFERENCE]` to
  observed fact with the green test file:line as evidence.
- [ ] **Step 6.2:** If GA is NO, the de-risk still succeeded — it identified the
  blocking layer. Pivot (§5 contract / Approach 2-3 / local fork) per doc §7.4,
  having spent days not weeks.

---

## 7. Regression sweep (before declaring done)

After A.2 and A.3, run representative existing tiling tests (R4 is per-commit;
this is the wider net):
```bash
for t in transform-op-fuse transform-op-tile multisize-tiling-full continuous-tiling-full; do
  .iree-build/llvm-project/bin/mlir-opt --transform-interpreter --split-input-file \
    third_party/llvm-project/mlir/test/Dialect/Linalg/$t.mlir \
    | .iree-build/llvm-project/bin/FileCheck \
      third_party/llvm-project/mlir/test/Dialect/Linalg/$t.mlir || echo "REGRESSION in $t"
done
```

---

## 8. Explicitly OUT OF SCOPE (future iteration)

- **`TilingInterface.td` contract change** — §5 fallback ONLY.
- **Initial-tiling writeback sites `:616`/`:447`** — not exercised by the
  motivating case (linalg initial tiling is unit stride; the strided writeback
  arises only in fusion). Touched for typedef consistency only if needed; untested.
- **Producer-fusion sites `:951`/`:1565` + rejection `:1502`** — symmetric to
  consumer fusion; deferred. (The typedef extension touches `:951` harmlessly
  with unit strides, but it is never exercised.)
- **IREE filter widening** (`TileAndFuseUtils.cpp:141`, `:154-155`) — pure
  consumer of this work.
- **Source-load co-distribution** (doc Q2) — IREE-side correctness for the real
  `m[0::2,0::2]=True` dispatch; separate change.
- **Arbitrary / coprime strides, affine stride composition, tile-size
  divisibility** (doc Q1/Q3) — static `2` only here.
- **#51660 vectorization** (doc §8.1) — performance, off the compile critical path.

---

## 9. Open questions for the next iteration pass

1. **(CENTRAL) Is the SCF-local offset/size composition correct?** A.2(c):
   stride-divide `allSizes` before `:2330`; post-multiply strided-dim offset
   after `:2347`. §3.1-B (`iv*2`) is the contract; EXEC-A is the judge. If
   correct → §5 skipped (Phase 0 ~1 day). **The single question Phase 0 exists
   to answer.**
2. **Stride-divide operator:** ceil-vs-floor division of span by stride, and
   whether `getIterationDomainTileFromOperandTiles`'s projected-permutation
   assumption holds once sizes are stride-divided. Resolve at impl; EXEC covers it.
3. **`mlir-cpu-runner` shared-libs / bufferization pipeline** (Step A.4):
   capture exact flags from an existing `mlir/test/Target/LLVMIR/` or
   `mlir/test/Integration/` execution test as template; don't invent.
4. **Whether sub-case B (offset scaling) needs a writeback-site change at all**
   — if tile-size-1 puts the stride entirely in the offset, B may be pure
   post-multiply with the per-tile writeback stride still `[2,1]`. Keep both
   §3.1 assertions regardless (R2).
5. **`YieldTiledValuesFn` typedef blast radius:** it is shared by consumer
   (`:1006`) and producer (`:951`) fusion. Confirm the extension is a pure
   addition (default unit) so producer fusion (`:1502`-gated) is unaffected.
6. **Upstream-vs-local-fork** is NOT decided here — Phase 0 makes it informed.
