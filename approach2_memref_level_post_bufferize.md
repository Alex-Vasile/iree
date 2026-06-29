# Approach 2 (NOTE §4-2) — Fuse at the MEMREF level (post-bufferization CPU distribution pass)

**Role of this document:** the CHEAPEST-CORRECTNESS-PATCH / stopgap ("Idea 2"). It determines exactly
what a CPU analogue of `GPUDistributeCopyUsingForallPass` would have to look like to wrap each bare
`linalg.generic` copy that comprehensive bufferization emits in a workgroup-mapped `scf.forall` so the
`VerifyWorkgroupDistributionPass` passes, and gives an independent verdict on whether to ship it as a
stopgap.

All IREE paths are relative to `compiler/src/iree/compiler/`. Per-pass IR is in the editor repro at
`/Users/alex/Developer/editor/rcd_lowpass_llvm_cpu_repro/dump.mlir`. `dump.mlir:NNNN` line refs are
that file. `[INFERENCE]` marks a reasoned (not directly-observed) step.

READ-ONLY investigation; no source edited, no builds run.

---

## 1. Verdict (one paragraph)

**Do-not-ship as-is; the "cheap" variant is silently INCORRECT, and the only CORRECT variant costs
about as much as the principled fix (Idea 1 / NOTE §4-1).** A CPU analogue of the GPU pass is
mechanically easy to sketch (match a copy-shaped `linalg.generic`, emit a workgroup-mapped
`scf.forall`, insert it right after `addCPUBufferizePasses`), and it WOULD silence the verifier. But
the obvious degenerate design — wrap each copy in a single-workgroup `[0,1)` forall so it runs in
workgroup 0 only — produces **wrong output**: the fill `forall` is itself workgroup-distributed, so
in workgroup 0 only tile (0,0) of the scratch is filled, and the gather/scatter copies (pinned to
workgroup 0) then propagate a result where only one tile's worth of `True` reaches the output. The
only *correct* memref-level fix must tile the strided gather/scatter copies across the same workgroup
grid as the fill (so each workgroup gathers→fills→scatters its own tile), which is exactly the
2-D-strided-dilation problem that is the substance of Idea 1 — i.e. the stopgap stops being cheap the
moment it is made correct. Independently of the perf/scratch problems the prior review raised, the
correctness trap alone rules out shipping a naive version; and a *correct* version has no compelling
advantage over Idea 1. **Recommendation: do-not-ship; if a temporary unblock is truly required before
Idea 1 lands, prefer the model-side workaround (expert_review Idea 4) over this.**

---

## 2. The reference GPU pass, distilled — and its three mismatches vs. the CPU need

### 2a. What the pass actually does
`Codegen/Common/GPU/GPUDistributeCopyUsingForall.cpp` (read in full, 153 lines):

- **Match:** `memref::CopyOp` only. The `runOnOperation` walk (`:120-151`) collects copies with a
  `PreOrder` walk that `skip()`s the children of any `scf.forall` carrying a **thread/warp/lane**
  mapping (`forallOpHasMappingType<IREE::GPU::LaneIdAttr, gpu::GPUWarpMappingAttr,
  gpu::GPUThreadMappingAttr>` at `:132-134`) and then `dyn_cast<memref::CopyOp>` at `:138`.
- **Distribute:** for each copy it calls `getCopyTileSizes(rewriter, copy)` (`:147`) and
  `distributeCopyToThreads(rewriter, copy, tileSizes)` (`:148`).
- **Body generator** (`distributeCopyToThreads`, `:47-112`): builds an `scf.forall` over the copy's
  full range (`upperBounds = memref::getMixedSizes(source)`, `:61-62`), strides = tile sizes, with a
  **thread** mapping array (`gpu::GPUThreadMappingAttr` LinearDim0..N, `:64-72`), then inside the
  body materializes `memref::SubViewOp` of source and target at the forall offsets/sizes (`:107-110`)
  and `rewriter.replaceOpWithNewOp<memref::CopyOp>(copy, sourceTile, targetTile)` (`:111`). The
  degenerate single-thread fallback is `distributeCopyToSingleThread` (`:33-44`): a 1-iteration
  forall with `MappingId::LinearDim0`.
- **Tile sizes** (`getCopyTileSizes`, `Codegen/Utils/Utils.cpp:2314-2325`): all dims = 1 except the
  innermost, which is `kPreferredCopyNumBits / elementBitWidth` = `128 / bw` (`Utils.cpp:56`). So it
  copies 128-bit columns per thread. (There is a separate `linalg::CopyOp` overload at
  `Utils.cpp:2327+`, irrelevant here.)

The pass is a textbook "lift a copy into a mapped `scf.forall` with per-tile subviews." Its skeleton
*is* the right shape to borrow. It is the three attachments that are wrong for the CPU case.

### 2b. Wiring (and a correction to the prior review)
`createGPUDistributeCopyUsingForallPass` is wired into **exactly one** backend pipeline:
`Codegen/LLVMGPU/Passes.cpp:592` (Step 8, immediately after `addGPUBufferizePasses` at `:588`, and
followed by `createGPUVerifyDistributionPass()` at `:596`).

**Correction to `expert_review.md:266-273`:** that review claims the pass is also wired into the
SPIR-V pipelines at `SPIRV/Passes.cpp:313, 429, 526`. That is wrong. Those three lines are
`createGPUDistributeSharedMemoryCopyPass` (`SPIRV/Passes.cpp:313, 429, 526` — a *different* pass, the
older `GPUDistributeSharedMemoryCopyPass`), **not** `createGPUDistributeCopyUsingForallPass`. A direct
read of `SPIRV/Passes.cpp` shows zero occurrences of `DistributeCopyUsingForall`. I also confirmed by
directory-wide grep that `createGPUDistributeCopyUsingForallPass` / `GPUDistributeCopyUsingForallPass`
appears in exactly one `.cpp` (its own) plus its `.td` declaration, and one wiring site
(`LLVMGPU/Passes.cpp:592`). It is **absent from every `LLVMCPU` pipeline** (see §4 for the CPU
pipeline). The review's *conclusion* (no CPU pass exists to reuse) is correct; only its citation of
SPIR-V wiring is not.

Note also that SPIR-V's own `gpuCopyFn` (`SPIRV/Passes.cpp:91-110`) emits `memref::CopyOp` *directly*
(`memref::CopyOp::create` at `:105`), which is why a `memref::CopyOp`-matching pass is coherent on
SPIR-V; it is then lowered to `linalg.generic` later by `createMemrefCopyToLinalgPass`
(`SPIRV/Passes.cpp:168`, builder in `Codegen/Common/MemrefCopyToLinalg.cpp:20-34`). The CPU copy fn
takes a different path (§3).

### 2c. The three mismatches (all three must change for a CPU analogue)

| Axis | GPU pass (as built) | CPU need (this bug) |
|---|---|---|
| **Op matched** | `memref::CopyOp` (`:138`) | `linalg::GenericOp` — a *copy-shaped* generic (identity maps, 1 in/1 out, `yield %in`). The llvm-cpu copies are emitted as `linalg.generic` directly, never as `memref.copy` (§3). |
| **Mapping emitted** | thread/warp/lane: `gpu::GPUThreadMappingAttr` etc. (`:69`, `:132-134`) | **workgroup**: `IREE::Codegen::WorkgroupMappingAttr`. The verifier (`VerifyWorkgroupDistribution.cpp:30-32`) demands a workgroup (or split-reduction) mapped forall; thread mappings do not satisfy it. |
| **Skip predicate** | skip children of thread/warp/lane foralls (`:132-134`) | skip children of **workgroup** foralls (mirror the verifier's own predicate, `VerifyWorkgroupDistribution.cpp:30-32`). Otherwise the pass would re-wrap copies already inside the fill forall. |

(The "wiring" difference — GPU-only, absent from CPU — is the fourth axis, fixed simply by adding a
CPU pipeline insertion; see §4.)

---

## 3. Op-shape mismatch in detail: CPU emits `linalg.generic`, not `memref.copy`

### 3a. How the CPU copies are born
llvm-cpu bufferizes via `addCPUBufferizePasses` (`Codegen/Common/CPU/Passes.cpp:46-50`), which hands
`cpuCopyFn` (`:40-44`) to `addIREEComprehensiveBufferizePasses` as `options.memCpyFn`. `cpuCopyFn`
calls `createLinalgCopyOp(builder, loc, from, to)` (`CPU/Passes.cpp:42`).

`createLinalgCopyOp` (`Codegen/Utils/Utils.cpp:2287-2312`) builds a `linalg::GenericOp`:
- inputs = `from`, outputs = `to` (`:2304-2305`);
- `indexingMaps = [multi-dim-identity, multi-dim-identity]` (`:2298-2299`, `:2306`);
- `iteratorTypes = all parallel` (`:2300-2301`, `:2307`);
- body = `linalg::YieldOp::create(..., args.front())` (`:2308-2310`) — i.e. `yield %in`.

So **the op to match on CPU is `linalg::GenericOp` with a copy signature**, not `memref::CopyOp`.
The default `defaultMemCpyFn` (`IREEComprehensiveBufferize.cpp:73-77`) produces the *same* op shape;
the prior review's note that llvm-cpu uses `cpuCopyFn` rather than `defaultMemCpyFn` changes only the
symbol, not the emitted op (`expert_review.md:88-100`).

### 3b. `MemrefCopyToLinalg.cpp` goes the *wrong* way for reuse
`Codegen/Common/MemrefCopyToLinalg.cpp:20-34` is an `OpRewritePattern<memref::CopyOp>` that calls
`createLinalgCopyOp` to turn `memref.copy` **into** `linalg.generic`. The two passes' op vocabularies
do not overlap in the direction we need: the GPU pass matches `memref.copy`, the CPU copies are
already `linalg.generic`.

### 3c. Design decision: match `linalg.generic` directly
**Decision: match the copy-shaped `linalg::GenericOp` directly.** Do **not** first canonicalize the
copies to `memref.copy` to reuse the GPU body. Reasons:

1. There is no upstream `linalg.generic → memref.copy` canonicalization pattern in this codebase
   (`MemrefCopyToLinalg.cpp` is one-way), so "canonicalize then reuse" would itself need a new
   pattern — more code, not less.
2. The body generator is trivial to re-derive for `linalg.generic`: the GPU pass's
   `memref::SubViewOp` + `memref::CopyOp` body (`GPUDistributeCopyUsingForall.cpp:107-111`) becomes
   `memref::SubViewOp` (source) + `memref::SubViewOp` (target) + a copy-shaped `linalg::GenericOp`
   (or simply reuse `createLinalgCopyOp`, `Utils.cpp:2287`).
3. Matching `linalg.generic` requires a precise predicate to avoid mis-matching *compute* generics
   (the fill anchor is itself a `linalg.generic`!). The fill is distinguished by having **no inputs**
   and `yield`ing a constant (`dump.mlir:31665-31667`); a copy has **1 input, 1 output, identity
   maps, all-parallel, `yield %in`**. The match predicate:
   - `linalg::GenericOp` with `getNumDpsInputs() == 1` and `getNumDpsInits() == 1`;
   - both `indexing_maps` are the multi-dim identity (rank R);
   - all `iterator_types` parallel;
   - the single region block has exactly one op other than the terminator, and the terminator is
     `linalg.yield` of the block argument corresponding to the input (i.e. `yield %in`).

   This uniquely identifies copies and excludes fills/reductions.

---

## 4. Mapping-level mismatch + pipeline insertion point (concrete design)

### 4a. Mapping to emit
The verifier requires the forall's mapping to begin with a `IREE::Codegen::WorkgroupMappingAttr` (or
`SplitReductionMappingAttr`) — `forallOpHasMappingType` (`Codegen/Utils/GPUUtils.h:62-70`) only inspects
`*mapping.value().begin()` (the *first* mapping attr). Construct it exactly as the tiler does in
`getMapping` (`Codegen/Common/TileDispatchUsingForall.cpp:145-167`):

```cpp
IREE::Codegen::WorkgroupMappingAttr::get(
    context, IREE::Codegen::symbolizeWorkgroupId(dim).value())
```
where `dim` ∈ {0→IdX, 1→IdY, 2→IdZ} (`TileDispatchUsingForall.cpp:158-159`). For a 2-D copy, emit
`[IdY, IdX]` reversed to match row-major workgroup ordering (the tiler reverses at `:149` and `:166`).

### 4b. Tile-size policy — and the correctness trap (the central finding)

**The GPU policy does not transfer to CPU.** On GPU the copy is tiled across the *thread* grid (a
fixed warp/lane grid), and `getCopyTileSizes` (`Utils.cpp:2314-2325`) picks 128-bit columns per
thread. On CPU there is no thread grid at the verifier's level — only the *workgroup* grid — and the
copy's tiling must be **consistent with the dispatch's single workgroup count**, or the copy either
runs redundantly in every workgroup (race/waste) or in only one workgroup (incomplete). This is where
a naive stopgap goes wrong:

#### The degenerate `[0,1)` forall is INCORRECT (do not do this)
The tempting "always-safe" design is the direct analogue of `distributeCopyToSingleThread`
(`GPUDistributeCopyUsingForall.cpp:33-44`): wrap each stray copy in a 1-iteration, workgroup-mapped
`scf.forall`. The verifier would pass (the copy is now lexically inside a workgroup forall). **But
the resulting program is wrong.** Grounding:

1. A workgroup-mapped `scf.forall` lowers to a *per-workguard* loop: `scf.for %iv =
   workgroup_id*step to ub step workgroup_count*step` (directly visible in
   `Codegen/LLVMCPU/test/vector_lowering.mlir:181-186` distributing a strided output via
   `affine.apply workgroup_id*4096`, and `Codegen/LLVMCPU/test/gpu_reorder_workgroups.mlir:14-19`;
   the ABI load is `loadWorkgroupID`/`loadWorkgroupCount`, `Codegen/LLVMCPU/DispatchABI.cpp:658-678`,
   converted in `ConvertToLLVM.cpp:239-251`). A `[0,1)` forall therefore lowers to
   `scf.for %iv = workgroup_id to 1 step workgroup_count`, which executes **only when
   `workgroup_id == 0`** — i.e. only in workgroup 0.
2. The scratch buffers are **per-workgroup stack**: `%alloca = memref.alloca()` (`dump.mlir:31791`)
   and `%alloca_3 = memref.alloca(%dim, %dim_2)` (`dump.mlir:31774`) — each workgroup has its own.
3. The fill `forall` is **workgroup-distributed**: workgroup (i,j) fills *only* its own 64×64 tile of
   its own scratch (`dump.mlir:31652-31677`, mapping `[workgroup y, workgroup x]`,
   `dump.mlir:31677`).

Tracing the wrapped program in workgroup 0: gather #1 (`dump.mlir:31736`, input-region→scratch1)
runs only in WG0 and fills *all* of WG0's scratch1 with the input; the fill forall in WG0 fills only
tile (0,0) of scratch1 with `True`; scatter #1 (`:31780`) reads WG0's scratch1 (= input with only
tile(0,0) overwritten by `True`); scatter #2 (`:31784`) writes that to the output binding. **Result:
the output receives `True` only in tile (0,0)'s corner pattern; every other tile's corners retain the
input value instead of `True`.** The intended `m[0::2,0::2]=True` is satisfied for only one 64×64
tile. Silently wrong, and *not* caught by the verifier (which is structural, not semantic —
`VerifyWorkgroupDistribution.cpp` does no address-overlap analysis). This is strictly worse than the
status quo (a clear compile error): it would ship a miscompilation.

> This is the key correction to the prior review. `expert_review.md:176-194` judges Idea 2
> "PARTIALLY SOUND … cheapest correctness patch" and lists its downsides as perf/scratch/masking —
> implicitly assuming the wrap is *correct*. The degenerate wrap is not correct; "satisfies the
> verifier" ≠ "computes the right answer."

#### The correct tiling ≈ Idea 1's complexity
To be correct, each workgroup must gather→fill→scatter **its own tile**, with the gather/scatter
tiles aligned to the fill tiles. That requires the stopgap to:
- recover the workgroup tile structure post-bufferization (still available: the inside-forall fill
  retains `lowering_config = #iree_cpu.lowering_config<distribution = [64, 64],
  vector_common_parallel = [1, 4]>`, `dump.mlir:31753`; equivalently the fill forall's step,
  `dump.mlir:31740` step 64);
- generate, per stray copy, a workgroup-mapped `scf.forall` whose body produces a `memref.subview` of
  the (strided) source and (strided) target at the workgroup's tile offsets, and a copy-shaped
  `linalg.generic` between them;
- handle the **stride mismatch**: gather reads a dim-1-strided region (`%subview_0 … strided<[?,2]>`,
  `dump.mlir:31733`), scatter writes a dim-0-strided output (`%subview … strided<[?,1]>`,
  `dump.mlir:31731`), and the fill tiles the dense `[H, W/2]` region. Aligning a per-workgroup
  subview of a *strided* memref to the fill's *dense* tile grid is the 2-D-dilation subview
  generation that NOTE §3 calls out as the substance of the proper fix.

That is precisely the affine-dilation capability Idea 1 / NOTE §4-1 adds at the tiler. Re-deriving it
post-bufferization, on already-strided memrefs, is *harder* (the tiler has clean tensor strides; here
we reconstruct from memref layouts), not easier. The "cheap stopgap" stops being cheap the moment it
is made correct.

> `[INFERENCE]` There is also an open mechanical question of whether the CPU workguard lowering
> tolerates **multiple** sibling workgroup-mapped foralls in one dispatch. The Transform-dialect
> distributor explicitly assumes "the **unique** topLevel `scf.forall`"
> (`TransformExtensions/CommonExtensionsOps.td:323-329`); the produced lowered form in
> `vector_lowering.mlir` / `gpu_reorder_workgroups.mlir` uses a single `workgroup_id`/`workgroup_count`
> pair per dim shared by all loops, which mechanically supports multiple foralls, but a correct tiled
> stopgap would add ≥4 more workgroup foralls alongside the fill, and each would need to share the
> identical grid. This is not verified here; it is an additional risk a correct design must discharge.

### 4c. Pipeline insertion point
The stopgap must run **after** `IREEComprehensiveBufferizePass` (which emits the `linalg.generic`
copies) and **before** `VerifyWorkgroupDistributionPass`. On llvm-cpu the verifier is a *module*-level
pass added in `buildLLVMCPUCodegenPassPipeline` at `Codegen/LLVMCPU/Passes.cpp:706`, immediately after
`createLLVMCPULowerExecutableTargetPass` (`:702-705`). The bufferization happens *inside* that target
pass, via the func-level expert pipelines (e.g. `addMultiTilingExpertPassPipeline` = the
`DoubleTilingExpert` used by this repro, `LLVMCPU/Passes.cpp:182-280`, which calls
`addCPUBufferizePasses` at `:266`).

Cleanest single insertion point: **inside `addCPUBufferizePasses` itself**
(`Codegen/Common/CPU/Passes.cpp:46-50`), immediately after
`addIREEComprehensiveBufferizePasses(funcPassManager, allocationFn, memcpyFn)` (`:49`). That single
edit covers all six expert-pipeline call sites (`LLVMCPU/Passes.cpp:266, 324, 389, 436, 477, 494`)
plus the registered `LLVMCPUVectorLoweringPipeline` (`:845`) automatically, and runs before the
module-level verifier. A pass that no-ops when the dispatch has no workgroup forall (mirror the
verifier precondition, `VerifyWorkgroupDistribution.cpp:29-43`) is safe in single-workgroup
pipelines too.

### 4d. Sketched IR, before → after (correct, tiled variant)

Before (the verifier's view; `dump.mlir:31736-31784`, abridged). Four bare copies at dispatch scope,
one workgroup forall (the fill) between gather #1 and the scatters:
```mlir
%subview   = memref.subview %33[0,0][%29,%30][2,1]            // OUTPUT, dim-0 strided            (31731)
%alloca    = memref.alloca() : memref<…xi8, #hal.descriptor_type<storage_buffer>>               // scratch1 (31791)
%subview_1 = memref.subview %alloca[0,0][%29,%34][1,1]                                          // scratch1 region
linalg.generic ins(%subview_0 : …<strided<[?,2]>>) outs(%subview_1 : …<strided<[…,1]>>)          // gather#1 (31736) ← BARE
scf.forall (…) … { …fill… } {mapping=[wg y, wg x]}                                              // fill     (31740)
%alloca_3  = memref.alloca(%dim,%dim_2) : …                                                     // scratch2 (31774)
linalg.generic ins(%32 : …<storage_buffer>) outs(%alloca_3 : …<storage_buffer>)                  // gather#2 (31775) ← BARE
linalg.generic ins(%subview_1 : …) outs(%subview_4 : …<strided<[?,2]>>)                          // scatter#1(31780) ← BARE
linalg.generic ins(%alloca_3 : …) outs(%subview : …<strided<[?,1]>>)                             // scatter#2(31784) ← BARE  [writes OUTPUT]
```

After (correct tiled stopgap — each bare copy becomes a workgroup forall whose per-WG body subviews
the strided source/target at the WG tile and copies that tile; only scatter#2 sketched):
```mlir
// distribution tile [64,64] recovered from the fill forall / its lowering_config
scf.forall (%wy, %wx) = (0,0) to (%29, %30) step (64,64)
    {mapping = [#iree_codegen.workgroup_mapping<y>, #iree_codegen.workgroup_mapping<x>]} {
  %ro = affine.min …(%wy)            // 64 (workguard-style bounds; same shape as vector_lowering.mlir:184)
  %co = affine.min …(%wx)
  %srcTile = memref.subview %alloca_3[%wy*64?, %wx*64?] [%ro, %co] [1,1]    // dense scratch2 tile
  %dstTile = memref.subview %subview   [%wy*1,  %wx*1 ] [%ro, %co] [2,1]    // OUTPUT, dim-0 strided — the dilation
  createLinalgCopyOp(%srcTile, %dstTile)                                    // copy-shaped generic, now INSIDE wg forall
}
```
The `strided<[?,1]>` output subview with `[2,1]` strides is exactly the 2-D-dilation write that
NOTE §3 "per-workgroup distribution" requires; producing it correctly for *all four* copies (two of
which are dim-1-strided gathers) is the work. The degenerate `[0,1)` variant would *also* type-check
and *also* pass the verifier — which is why the trap is easy to fall into and hard to catch in review.

---

## 5. NOTE §4 open questions — answered with citations

**Verifier interaction (§4 "Interaction with the verifier").** Confirmed: the verifier is a `PreOrder`
walk (`VerifyWorkgroupDistribution.cpp:48`) that `skip()`s all children of a workgroup-mapped
`scf.forall` (`:49-55`), then flags any `MemoryEffectOpInterface` op with a `Write` effect on a global
memref operand (`:57-76`). "Global" = `hasGlobalMemoryAddressSpace` (`:68`), which returns true for
`#hal.descriptor_type<storage_buffer>` via `isa<IREE::HAL::DescriptorTypeAttr>`
(`Codegen/Utils/GPUUtils.cpp:1221`). So lexically nesting each bare copy inside a workgroup forall
makes the walk `skip()` it (`:54`) and the verifier passes. **The four bare copies in the verifier's
view** (`dump.mlir:31683-31789`, `IR Dump After IREEComprehensiveBufferizePass`):

| # | Line | Op | src → dst | dst is | a stopgap must wrap? |
|---|------|----|-----------|--------|----------------------|
| 1 | `:31736` | `linalg.generic` | `%subview_0` (input, dim-1 strided) → `%subview_1` | scratch1 region | yes — writes a `storage_buffer` scratch |
| 2 | `:31775` | `linalg.generic` | `%32` (input binding) → `%alloca_3` | scratch2 | yes — writes a `storage_buffer` scratch |
| 3 | `:31780` | `linalg.generic` | `%subview_1` (scratch1) → `%subview_4` | scratch2 (dim-1 strided) | yes — writes a `storage_buffer` scratch |
| 4 | `:31784` | `linalg.generic` | `%alloca_3` (scratch2) → `%subview` | **OUTPUT binding** (dim-0 strided) | **yes — irreducible** |

All four write a `#hal.descriptor_type<storage_buffer>` memref, so all four trip the verifier; a
stopgap must wrap all four to silence it. Only #4 is a *genuine* output-binding write
(`expert_review.md:123-131`, Gap 2): `%subview` is a `subview` of the output HAL binding `%33`
(`dump.mlir:31731`). #1–#3 write scratch, and would drop off the verifier's radar if the scratch lost
its descriptor memory space (see §5 "memory-space"). **All four must be wrapped; distinguishing #4
only matters for understanding why no scratch trick alone can fix this.**

**Read-modify-write / scratch (§4 "Zero-preservation / read-modify-write").** Confirmed: the
post-bufferize body keeps the full redundant gather + scratch chain — `%alloca` (scratch1,
`dump.mlir:31791`, statically `2^52×2^52` from the `util.assume.int` umax, `dump.mlir:31720-31724`)
and `%alloca_3` (scratch2, dynamic, `dump.mlir:31774`); gather#1 (`:31736`), gather#2 (`:31775`),
scatter#1 (`:31780`), scatter#2 (`:31784`). The dispatch is read-modify-write: it loads the whole
input (`load_from_buffer %32`, `dump.mlir:31650`) to preserve non-scattered cells. A memref-level
stopgap does **not** touch this — it wraps the existing copies, so the redundant gather/scatter and
both scratch allocas remain. **Perf cost** (qualitative, `[INFERENCE]`): every dispatch pays for two
full-size gathers + two full-size scatters + the scratch traffic, all of which a proper in-place
distribution (Idea 1) eliminates by aliasing the fill to the output. For an `[H,W]` boolean mask
this is on the order of 4× the binding I/O plus a scratch the size of the binding — a large constant
overhead, and for the static-`2^52` scratch it is not merely slow but separately illegal (see next).

**Memory-space leak (§4 / `expert_review.md` Gap 1).** Confirmed: `cpuAllocationFn`
(`Codegen/Common/CPU/Passes.cpp:21-38`) forwards the requested `MemRefType` *verbatim* to
`memref::AllocaOp::create` (`:35-37`) with **no** descriptor-space erasure, unlike
`defaultAllocationFn` (`IREEComprehensiveBufferize.cpp:56-72`) which explicitly erases it at `:61-69`
("We cannot allocate to generate a resultant MemRef type with descriptor type memory space … So erase
and fallback to the default 0 memory space"). One-shot bufferization derives the allocation type from
the source tensor preserving `memorySpace`, so a scratch that is a view of a HAL binding carries the
descriptor space, and the CPU allocator honours it. **Answer to the open question:** *wrapping the
copies in workgroup foralls is sufficient to silence the verifier* (it is purely lexical), so a
stopgap need NOT also fix the scratch memory space. **But** the static `memref<2^52 × 2^52>` scratch
(`dump.mlir:31791`) remains a latent, independent defect: once the verifier is satisfied, that alloca
would trip `max_stack_allocation_size` (the llvm-cpu target attr, `expert_review.md:235-251`, Idea 5).
So a stopgap that only wraps foralls would unblock the verifier and then **immediately** hit the
stack-size guard — i.e. it does not even produce a linkable dispatch for this repro. That is a second,
independent reason the stopgap is inadequate as shipped.

---

## 6. What this approach does NOT fix (explicit)

1. **The root defect (flow + tiling).** The transposed read-modify-write dispatch is born at flow
   formation (fill region dim-1-strided `insert_slice [1,2]`, output store dim-0-strided
   `strided<[?,1]>`; `dump.mlir:31651` vs `:31648`) and survives because the tiler strands the store
   outside the forall (`fuseConsumersIntoForall` only fuses `ParallelInsertSliceOp`,
   `Codegen/Common/TileAndFuseUtils.cpp:141-144`; `store_to_buffer` is not even a compute op). A
   post-bufferize wrap changes none of this — the malformed dispatch is still formed and still tiled
   the same way; future transposed dispatches hit the same wall.
2. **Performance.** The redundant gather×2 + scatter×2 + scratch traffic remain (§5). A wrapped copy
   is a *sequential* copy moved into a forall, not the in-place alias Idea 1 achieves.
3. **The `2^52 × 2^52` scratch and `max_stack_allocation_size`.** Untouched (§5). The stopgap does
   not even reach a clean link for this repro.
4. **Correctness (naive variant).** The degenerate `[0,1)` wrap miscompiles (§4b). Only the
   ~Idea-1-cost tiled variant is correct.

---

## 7. Recommendation + guardrails

**Recommendation: do-not-ship.** Three independent reasons, any one sufficient:

1. **The cheap variant is silently incorrect** (§4b): a degenerate wrap passes the verifier but
   produces wrong output (only tile (0,0) of the mask is written), and the verifier cannot catch it.
   Shipping a miscompilation to unblock a test is strictly worse than a clear compile error.
2. **The correct variant is not cheap** (§4b): it must generate per-workgroup strided subviews
   aligned to the fill grid across two different stride axes — i.e. re-implement the core of Idea 1,
   on harder (already-bufferized, strided-memref) input. There is no effort/risk advantage left.
3. **It does not reach a linkable dispatch for the repro** (§5): the `2^52 × 2^52` scratch trips
   `max_stack_allocation_size` immediately after the verifier is satisfied, so it fails one step
   later anyway.

This sharpens `expert_review.md:176-194` ("PARTIALLY SOUND … Do not ship as the fix"): the review's
*conclusion* (don't ship) is right, but its *premise* (the wrap is a correct, cheap patch) is not —
the correctness failure makes the case against shipping stronger than "it masks the defect."

**If a temporary unblock is genuinely required before Idea 1 lands**, in preference order:
- **Model-side workaround** (`expert_review.md:223-232`, Idea 4): replace the out-of-place
  `m[0::2,0::2]=True` with an in-place `fill_(True)` on the strided view, which lowers to a
  single-axis-strided in-place store that already distributes correctly (`control_dump.mlir`). Zero
  compiler risk, <1 h, correctly scoped, paired with an upstream issue. **Preferred.**
- Only if a compiler-level unblock is mandated AND Idea 1 is far off, consider a **heavily-guardrailed**
  stopgap, with ALL of:
  - **Correctness:** tile the copies across the workgroup grid aligned to the fill (never the
    degenerate `[0,1)` wrap). Add a lit test asserting the full mask (not one tile) is written for a
    small static `[H,W]`, and a runtime numerical check in the editor test.
  - **Scope guard:** match *only* the transposed-strided pattern (a `linalg.generic` copy whose
    target is a strided subview of a HAL binding and whose source/tile grid is recoverable from a
    sibling workgroup forall), not arbitrary copies — to avoid disturbing the many dispatches that
    already compile.
  - **Observability:** emit a `remark` ("strided output store distributed post-bufferization; "
    "dispatch was formed transposed — tracked under <issue>") so the masked defect stays visible.
  - **Lifetime:** land behind a flag defaulting off, with a removal issue tied to Idea 1.
  - **Pre-conditions:** skip (no-op) when the dispatch is single-workgroup or when the scratch is
    statically oversized (to avoid trading a verifier error for a stack-size error).

Even with all guardrails, this is engineering time better spent on Idea 1 / NOTE §4-1 (or the
flow-formation fix, NOTE §4-3), which fix the defect at its source for all backends at once.
