# Synthesis — the three NOTE §4 approaches, compared

**Purpose.** This is the capstone synthesis over three independent, source-grounded
subagent investigations, one per approach in
`strided_store_workgroup_distribution_investigation.md` §4:

| NOTE §4 | Approach | Subagent findings file |
|---|---|---|
| §4-1 | Fuse at the **tensor** level (extend the tiler / `fuseConsumersIntoForall`) | `approach1_tensor_level_fusion.md` |
| §4-2 | Fuse at the **memref** level (post-bufferization CPU pass, GPU analogue) | `approach2_memref_level_post_bufferize.md` |
| §4-3 | **Reuse the GPU distribution strategy** + "is anything similar being done?" survey | `approach3_gpu_strategy_reuse_survey.md` |

All three subagents ran read-only (no source edits, no builds), grounded every claim in a
`file:line` they personally opened, and tagged reasoned steps `[INFERENCE]`. This synthesis
adds nothing not supported by those files or by my own reproduction; `[INFERENCE]` is
inherited where noted.

> **Taxonomy note.** These three are the NOTE's own §4 split. They do **not** map 1:1 onto
> the editor-repo `OVERVIEW.md` "Idea 1/2/3" numbering. The flow-formation fix (OVERVIEW
> "Idea 3" / the NOTE's "distinct from Idea 3") is **out of scope** for this dispatch; it is
> referenced only where it bears on the recommendation.

---

## 1. The bug, in one diagram (reproduced by the main agent 2026-06-29)

`m[0::2, 0::2] = True` → flow fuses two axis-disjoint scatters into one **transposed**
read-modify-write dispatch. The defect is layered — born at flow formation, propagated by
tiling, materialized by bufferization, detected by the verifier:

```
flow formation      tiling                  bufferization              verifier
────────────────────────────────────────────────────────────────────────────────────
fill region:  [1,2]   tiler tiles ONLY       cannot alias [1,2]         every write to a
(dim-1 strided)       the fill into the      region into [2,1]          #hal.descriptor_type
output store: [2,1]   workgroup forall;      output → emits BARE        <storage_buffer> memref
(dim-0 strided)       STRANDED store         linalg.generic copies      must be lexically INSIDE
  ^^^ axis        outside the forall         OUTSIDE the forall         a workgroup-mapped
  MISMATCH                                   (4 of them)                scf.forall  →  REJECT
```

**Reproduction (main agent, this session):**
`iree-compile --iree-hal-target-backends=llvm-cpu reduced_reproducer.mlir -o /dev/null`
→ exit 1 with **both** oracle substrings:
- `'linalg.generic' op write affecting operations on global resources are restricted to workgroup distributed contexts.`
- `'func.func' op failed on workgroup distribution verification`

The offending dispatch body is directly visible: output subview `static_strides=[2,1]`
(`strided<[?,1]>`), input subview `static_strides=[1,2]` (`strided<[?,2]>`) — the axis
mismatch is in the IR. A `memref<4503599627370496 x 4503599627370496>` scratch carries
`#hal.descriptor_type<storage_buffer>` (the `cpuAllocationFn` memory-space leak, expert_review
Gap 1). One workgroup `scf.forall` (the fill into scratch); four bare `linalg.generic`
copies outside it, the last (`%47 → %43`) being the irreducible genuine output-binding write.

**Backend-general** — confirmed by Approach 3 reproducing `metal-spirv` (exit 1) too.

---

## 2. Cross-cutting truths (all three investigations converge)

These are the load-bearing findings. Each is independently corroborated across at least two
of the three files.

### 2.1 No dilation primitive exists in IREE codegen — the capability is net-new
NOTE §4's open question "is there an existing primitive for 'each workgroup writes every-k-th
element along a dimension'?" → **NO.** Approach 1 (§5 Q1) and Approach 3 (§B.6) independently
found the **same** bail sites, every one rejecting non-unit strides:
- `Common/CombineLayoutTransformation.cpp:220-223` — `non-unit strides are not supported`
- `Common/ReshapePatterns.cpp:415-416` & `:622-623` — `found a non-1 stride` / `expected unit strides`
- `Common/IREECodegenCanonicalizer.cpp:31-32` — folds unit-stride subviews only
- `Common/TensorDynamicDimAnalysis.cpp:110-113` — aborts unless flow-load strides are 1

The only `strides`/`dilations` attributes in the tree are **convolution window** attributes
(e.g. `linalg.conv_2d … {strides = dense<2>}`), consumed by conv-specific tiling — a semantic
property inside a structured op, not a workgroup-distributed-memory-write dilation. **The core
capability to add is an affine per-dimension dilation factor on the workgroup-tile → memory-offset
mapping.** It does not exist anywhere in `Codegen/`.

### 2.2 The single-stride control compiles via bufferization *aliasing*, NOT tensor fusion
The passing `m[:,0::2]` control is the closest precedent. Approach 1 (§3.2) and Approach 3
(§B.7) both opened `control_dump.mlir` and found: region strides `[1,1]` and output store
strides `[1,1]` are **identical on every axis**, so one-shot bufferization aliases the fill's
`shared_outs` in-place to the output binding, the fill writes straight into the global buffer
inside the workgroup `scf.forall`, and the degenerate `store_to_buffer` self-copy is
copy-eliminated. By the verifier's input there is **no write to a global outside the forall**.

> The control does **not** exercise tensor-level fusion of a strided store. Fusion was a no-op
> (the only outside-forall consumer was the non-`TilingInterface` `store_to_buffer`). It compiles
> purely because the unit-stride layouts alias. **The transposed two-axis case `[1,2]` vs `[2,1]`
> is what bufferization cannot alias — hence the bare copy.**

### 2.3 The proper fix is blocked by an UPSTREAM MLIR wall, not IREE's filter (Approach 1)
This is the single most important reframing. Widening IREE's
`fuseConsumersIntoForall` gate (`TileAndFuseUtils.cpp:141`) is ~10 lines and trivial — **but it
is a no-op** because upstream `mlir/lib/Dialect/SCF/Transforms/TileUsingInterface.cpp`
**structurally cannot represent a strided tensor write-back**:
- Hardcodes unit strides at **4** sites: `:447-451` (scf.for writeback), `:1006-1010` (forall
  parallel_insert writeback), `:2363-2367` (dest-extract), and the `resultStride` family.
- Explicitly **rejects** non-unit strides at **2** sites: `:2313-2317` (consumer fusion) and
  `:1502-1504` (producer fusion). `YieldTiledValuesFn` (`:1491-1495`) returns offsets/sizes
  only — no strides — so there is no plumbing to carry one.

Those rejections are correctness guards: removing them without the 4 hardcode fixes would
silently miscompile. The change must land **upstream** (IREE vendors
`third_party/llvm-project/`), and `TileUsingInterface.cpp` is foundational — every SCF-tiling
dialect (linalg, tensor, vector, affine, all of IREE) flows through it. Effort is therefore
**HARD+, >3 days, and political** (upstream acceptance), more than the prior review's "3d+".

### 2.4 The GPU machinery does NOT fix the bug — it relocates the defect (Approach 3)
Approach 3 reproduced `metal-spirv` (exit 1): the offending op is a `memref.store` to the
`#hal.descriptor_type<storage_buffer>` output binding inside **thread-distributed `scf.for`
loops**, **outside** the workgroup `scf.forall`, rejected by the **same Common
`VerifyWorkgroupDistributionPass`** the CPU path hits (wired as a module pass after
`createSPIRVLowerExecutableTargetPass`, `SPIRV/Passes.cpp:653-654`). NOTE §4 lines 86-88 are
confirmed accurate. The GPU pipeline stages through shared memory and distributes copies to
threads — but the final output storeback still never enters a workgroup forall. **GPU does not
absorb the stranded store; it relocates it into thread loops.** (The stricter `GPUVerifyDistributionPass`,
LLVMGPU-only, is never reached — the bug fails the Common verifier first.)

### 2.5 The borrowable GPU insight: stride is a property of the *view*, not the *loop*
Where the GPU path **is** instructive (Approach 3 §A.2/A.3): it never "dilates a loop." It
(1) bakes the stride into the destination memref **type** via a strided `memref.subview`, (2)
optionally stages through a dense shared-memory tile, and (3) emits a **unit-stride** load/store
loop over a dense tile that scatters into the strided global view. The cleanest expression is
`distributeCopyToThreads` (`GPUDistributeCopyUsingForall.cpp:47-112`): unit-stride subviews of
both operands (`strides(rank, getIndexAttr(1))`, `:106`) + a copy inside the loop; the
destination stride rides on `copy.getTarget().getType()`. **This shape transfers to a
workgroup-mapped forall verbatim** — only the `mapping` (thread → workgroup) changes. Shared-memory
staging has **no CPU analogue** and is a GPU coalescing optimization, not a correctness
requirement. (`distributeCopyToThreads` is stride-agnostic, so the borrowed shape generalizes to
arbitrary `[a,b]` strides and higher ranks for free.)

---

## 3. Approach-by-approach verdicts

### Approach 1 — tensor-level fusion (the proper fix). Verdict: SOUND but UPSTREAM-BLOCKED; the right *target*, the wrong *first move*.

- Direction correct: fusing the strided store into the workgroup `scf.forall` lets bufferization
  alias the fill to the output in-place (as the single-stride control does), no bare copies, the
  verifier passes naturally.
- The IREE-side change is small (widen `TileAndFuseUtils.cpp:141` to seed a strided
  `tensor.insert_slice`; relax the `filterFn` at `:154-155`). **But it is a no-op until upstream
  `TileUsingInterface.cpp` can carry strides** (§2.3).
- The two-axis case needs a tiled body that emits a strided `parallel_insert_slice`/store into the
  output binding (e.g. `[2,1]`), with the source load **co-distributed** (read-modify-write, not
  overwrite — the dispatch loads the whole input to preserve non-scattered cells).
- Verifier interaction confirmed benign: once the store is lexically inside the forall, the
  PreOrder walk skips it (`VerifyWorkgroupDistribution.cpp:48-56`); `shared_outs` in-place aliasing
  is legal (the control proves it); `verifyComputeOpsAfterDistribution` passes (the strided
  `tensor.insert_slice`, a `computeOp`, is now inside).
- Effort HARD+, >3d. Load-bearing difficulty is upstream, not the IREE pass.
- Recommended de-risk (cheap, in isolation): **fork `TileUsingInterface.cpp`, thread strides
  through `YieldTiledValuesFn`, replace the 4 unit-stride hardcodes + 2 rejections, add one
  `mlir/test/Dialect/SCF` lit test** fusing a `[1,2]`-strided `tensor.insert_slice` consumer into
  an `scf.forall`. This converts the central "can upstream represent this?" question from
  `[INFERENCE]` to a **binary yes/no** before any IREE plumbing. If `getResultTilePosition`'s
  affine model genuinely cannot express the stride, **Approach 1 is dead.**

### Approach 2 — post-bufferization CPU distribution pass (the stopgap). Verdict: DO-NOT-SHIP.

The cheapest variant is not just ugly — it is **silently incorrect**:
- The tempting "safe" design wraps each bare copy in a degenerate single-workgroup `[0,1)` forall
  (the analogue of `distributeCopyToSingleThread`, `GPUDistributeCopyUsingForall.cpp:33-44`). The
  verifier would pass. **But a workgroup-mapped forall lowers to a per-workguard loop
  `scf.for iv = workgroup_id*step to ub step workgroup_count*step`** (`vector_lowering.mlir:181-186`,
  `DispatchABI.cpp:658-678`), so a `[0,1)` forall runs **only in workgroup 0**. The fill `forall` is
  itself workgroup-distributed and the scratch buffers are per-WG stack allocas → only tile (0,0)
  of the mask gets `True`. **Miscompilation the verifier cannot catch** (it is structural, not
  semantic). This is strictly worse than the status quo (a clear compile error).
- The only **correct** variant must tile the strided gather/scatter copies across the same
  workgroup grid as the fill, with per-WG subviews of the strided source/target at the fill's tile
  offsets — i.e. re-implement Approach 1's 2-D dilation on **already-bufferized, strided memrefs**
  (harder, not easier). The stopgap stops being cheap the moment it is made correct.
- It does **not even reach a linkable dispatch** for the repro: the static
  `memref<4503599627370496 x 4503599627370496>` scratch trips `max_stack_allocation_size` one step
  after the verifier is satisfied.

Three independent reasons to not ship (any one sufficient). If a temporary unblock is truly
needed, prefer the **model-side `fill_(True)` workaround** (expert_review Idea 4) — zero compiler
risk, <1 h, correctly scoped.

### Approach 3 — reuse GPU strategy + survey. Verdict: CONTRIBUTES THE IR SHAPE; confirms no existing solution.

- Gives the proper fix the **concrete IR shape to target**: strided `memref.subview` + unit-stride
  inner tile + copy-inside-workgroup-forall (the tensor-level analogue is a strided
  `tensor.insert_slice` whose per-tile slice is unit-strided, fused into the workgroup forall).
- Proves (by metal-spirv reproduction) the GPU machinery **does not already fix it** — so this is
  not "an existing pass that should trigger but doesn't."
- The 16-row survey classifies every "distribute X into forall" pass by mapping level. **Only the
  workgroup-level passes satisfy `VerifyWorkgroupDistributionPass`**; the thread/warp/lane GPU
  passes do not (a thread loop is not a workgroup forall). No pass distributes a strided
  writeback into a workgroup forall.
- For a stopgap (Approach 2), `GPUDistributeCopyUsingForall` is the closest template but must be
  retargeted to match `linalg.generic` (CPU emits `linalg.generic`, not `memref.copy`) and emit
  `WorkgroupMappingAttr` — and a thread-loop stopgap would **still fail the Common verifier**.

---

## 4. Answers to the user's two cross-cutting questions

### 4.1 "Is anything similar to this currently being done in IREE?"
**No — for a strided workgroup-distributed writeback, nothing exists.** Evidence:
- **Survey (Approach 3 §B.5):** of every "distribute X into forall" pass in `Codegen/`, only the
  workgroup-level ones (`TileAndDistributeToWorkgroupsUsingForallOpPass`, `LLVMGPUTileAndDistribute`,
  `CombineLayoutTransformation`, `ReconcileTranslationInfo`, `ConvertWorkgroupForallToPCF`,
  `GPUPackPartialReductions`) produce a construct the verifier accepts — and the tiler's consumer
  fusion (`fuseConsumersIntoForall`, `:141`) only seeds `tensor::ParallelInsertSliceOp`, which is
  the exact filter that strands this store.
- **Closest relative, inapplicable on three independent axes:** `GPUDistributeCopyUsingForallPass`
  is (a) GPU-only (wired solely at `LLVMGPU/Passes.cpp:592` — Approach 2 corrects the prior review's
  erroneous SPIR-V citation), (b) matches `memref::CopyOp` while CPU emits `linalg.generic`, and
  (c) maps to threads/warp/lane, not workgroups.
- **GPU does not absorb it** (Approach 3 reproduced metal-spirv failing the same Common verifier).
- **No dilation primitive exists** (§2.1).

### 4.2 "How are similar distribution ops executed?"
Two reference points, both source-grounded:
- **Single-stride CPU control (compiles):** executes by **in-place aliasing** — one-shot
  bufferization aliases the fill's `shared_outs` to the output binding, the fill writes directly
  into the global buffer inside the workgroup `scf.forall`, and the degenerate `store_to_buffer`
  self-copy is eliminated. There is no separate store (`control_dump.mlir:25851-25889`,
  `:27459-27479`). This is the path the two-axis case fails to take because `[1,2]` ≠ `[2,1]`.
- **GPU strided copy:** executes by baking the stride into the **destination memref view**
  (`memref.subview static_strides`), staging through a dense shared-memory tile, and running a
  **unit-stride** load/store loop that scatters into the strided global view. The stride is a
  property of the *view*, not the *loop* (`GPUDistributeCopyUsingForall.cpp:104-112`). This is the
  shape a workgroup-level fix borrows.

---

## 5. Corrections to the prior `expert_review.md` (made by the subagents)

1. **SPIR-V wiring citation (Approach 2 §2b).** The review claims
   `createGPUDistributeCopyUsingForallPass` is wired at `SPIRV/Passes.cpp:313, 429, 526`. **Wrong.**
   Those lines are the *different* `createGPUDistributeSharedMemoryCopyPass`. A directory grep shows
   `DistributeCopyUsingForall` appears in exactly one wiring site (`LLVMGPU/Passes.cpp:592`). The
   review's *conclusion* (no CPU pass to reuse) stands; only its citation does not.
2. **Stopgap premise (Approach 2 §4b).** The review calls Idea 2 "PARTIALLY SOUND … cheapest
   correctness patch," implicitly assuming the wrap is *correct*. It is not: the degenerate wrap
   **silently miscompiles** (only WG0 runs the copy). "Satisfies the verifier" ≠ "computes the right
   answer." This makes the case against shipping *stronger* than "it masks the defect."
3. **Effort for the proper fix (Approach 1 §6).** The review rates Idea 1 "HARD (3d+)." Sharpened:
   the load-bearing difficulty is **upstream** `TileUsingInterface.cpp` (unit-stride hardcodes +
   rejections), not the IREE pass — so it is larger *and* political (upstream acceptance), >3d.

---

## 6. Recommendation & sequencing (decision tree)

```
                   Need to ship a fix?
                          │
        ┌─────────────────┴──────────────────┐
        NO (correctness/perf not blocking)    YES — temporary unblock needed now
        │                                     │
        Pursue the proper fix                 Model-side fill_(True) workaround
        (below).                              (expert_review Idea 4): <1h, 0 risk.
                                              Pair with an upstream issue.
                          │
        ┌─────────────────┴──────────────────────────────┐
        The proper fix has two viable homes:
        │
        (A) Approach 1 (NOTE §4-1) — distribute the strided store into the workgroup forall.
            Sound, backend-general, but UPSTREAM-BLOCKED.
            FIRST STEP (cheap, isolating): prototype the upstream TileUsingInterface.cpp
            stride-threading change + one mlir/test/Dialect/SCF lit test (Approach 1 §7).
            If the prototype proves the affine model can carry a stride → do the full
            IREE+upstream work. If it cannot → Approach 1 is dead; pivot to (B).
        (B) Flow-formation fix (OVERVIEW "Idea 3" — out of scope here, but the best
            proper alternative): do NOT fuse axis-disjoint scatters into one transposed
            dispatch. Two sub-options: emit two single-axis-strided stores (one extra
            dispatch, each compiles), or recognize the doubly-strided writeback and emit a
            single in-place store strides=[2,2]. Born at the layer where single-stride
            already makes the opposite choice.

        NOT RECOMMENDED under any framing:
        - Approach 2 stopgap: silently incorrect (cheap variant) or ≈ Approach 1 cost
          (correct variant), and fails one step later on the 2^52 scratch anyway.
        - Relaxing VerifyWorkgroupDistributionPass: it is correct and load-bearing (also for
          GPU correctness); the defect is a missing codegen capability, not an over-strict rule.
```

**Net:** Approach 1 is the right *architecture* and Approach 3 supplies its target IR shape, but
neither is a quick win — both run into the upstream SCF stride wall / the absence of any dilation
primitive. The honest near-term move is the model-side workaround; the honest long-term move is
**either** the Approach-1 upstream prototype (if it pans out) **or** the flow-formation fix. Do
not build Approach 2.

---

## 7. Artifacts

Findings files (this directory, `/Users/alex/Developer/iree/`):
- `approach1_tensor_level_fusion.md` — tensor-level fusion feasibility (619 lines, 25 sections)
- `approach2_memref_level_post_bufferize.md` — post-bufferization stopgap design + verdict (403 lines)
- `approach3_gpu_strategy_reuse_survey.md` — GPU strategy reuse + 16-row pass survey (345 lines)
- `synthesis_three_approaches_comparison.md` — this file

Repro & prior art: `/Users/alex/Developer/editor/rcd_lowpass_llvm_cpu_repro/`
(`reduced_reproducer.mlir`, `control_single_stride.mlir`, `dump.mlir`, `control_dump.mlir`,
`OVERVIEW.md`, `expert_review.md`). Binary: `/Users/alex/Developer/venv_iree/bin/iree-compile`.

Investigation source root: `/Users/alex/Developer/iree/compiler/src/iree/compiler/`
(paths in the findings files are relative to this). Upstream MLIR:
`/Users/alex/Developer/iree/third_party/llvm-project/mlir/`.
