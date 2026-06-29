# Approach 3 (NOTE §4-3) — Reuse the GPU distribution strategy + "is anything similar being done?" survey

*Staff-engineer findings. READ-ONLY investigation — every claim is grounded in a `file:line`
I personally opened; reasoned (not directly observed) claims are tagged `[INFERENCE]`. Reproductions
were run against the prebuilt `/Users/alex/Developer/venv_iree/bin/iree-compile` (observation only;
no source edits, no rebuild).*

> Scope note (per coordination with `Main`): this file covers ONLY (A) the GPU distribution strategy
> and its borrowability, and (B) the survey of every "distribute X into forall" pass + the
> dilation/stride-primitive question + how the single-stride control *executes* today. It does **not**
> do the tensor-level fusion *design* (generalizing single-stride → two-axis) — that is
> `Approach1TensorFusion`'s deliverable.

---

## 1. Verdict (one paragraph)

**The GPU strategy offers a clean, directly-borrowable IR *shape* for the proper fix — but not a
drop-in pass.** The reusable construct is exactly what `GPUDistributeCopyUsingForallPass`
(`Codegen/Common/GPU/GPUDistributeCopyUsingForall.cpp:104-112`) emits: a `scf.forall` whose body
takes **unit-stride subviews** of an already-strided source and target memref and runs a copy inside
the loop — the stride lives in the *memref type* of the destination (`strided<[?,1]>`), not in the
loop. That shape carries over to a *workgroup*-mapped forall verbatim; only the `mapping` attribute
(thread → workgroup) and the intermediate shared-memory staging (which has **no CPU analogue**) differ.
**However, the GPU machinery does *not* already fix this bug** — I reproduced `metal-spirv` (exit 1)
and the offending op is a `memref.store` to the `#hal.descriptor_type<storage_buffer>` output binding
inside *thread*-distributed `scf.for` loops, *outside* the workgroup `scf.forall`, rejected by the
**same** `VerifyWorkgroupDistributionPass` the CPU path hits. The GPU pipeline **relocates the defect
into thread loops, it does not absorb the stranded output store into a workgroup forall.** And the
answer to "is anything similar being done?" for *strided* writes is **no**: every stride-handling site
in the distribution/fusion machinery **rejects non-unit strides** (`CombineLayoutTransformation.cpp:220-223`,
`ReshapePatterns.cpp:415-416` & `:622-623`, `IREECodegenCanonicalizer.cpp:31-32`); the only
`strides`/`dilations` attributes in the tree are **convolution window** attributes, consumed by conv
tiling — there is **no** existing primitive for "each workgroup writes every-k-th element along a
dimension." That capability is the core thing to add.

---

## Part A — GPU distribution of copies/stores

### A.1 The GPU distribution pipeline, end to end

IREE has **two distinct GPU distribution models**, selected by backend. This matters because the bug
fails on `metal-spirv`, which uses the SPIR-V model — not the LLVMGPU one that the NOTE points at as
"the reference."

**Model 1 — LLVMGPU (CUDA/NVVM, ROCm/ROCDL): the `scf.forall` + thread-mapping model.**
Wired in `Codegen/LLVMGPU/Passes.cpp`. The relevant post-bufferization sequence (`LLVMGPU/Passes.cpp:591-600`):

| line | pass | role |
|---|---|---|
| `591` | `DecomposeMapStorePass` | lower `iree_linalg_ext.map_store` |
| `592` | **`createGPUDistributeCopyUsingForallPass`** | wrap each `memref::CopyOp` in a **thread**-mapped `scf.forall` |
| `593-595` | `NormalizeLoopBoundsPass` (forall only) | normalize forall bounds |
| `596` | **`createGPUVerifyDistributionPass`** | require writes in **thread/lane** contexts |
| `597` | **`createGPUDistributeForallPass`** | lower thread/warp-mapped foralls → `scf.for` + `gpu::ThreadIdOp` |
| `600` | `VectorizeMemrefCopyPass` | vectorize the copies |
| (module) `334` | `createGPUDistributePass` | map non-workgroup foralls to threads via upstream `mapOneForallToThreadsImpl` |

**Model 2 — SPIR-V (Metal, Vulkan): the `scf.for` + marker model + shared-memory-copy distribution.**
Wired in `Codegen/SPIRV/Passes.cpp`. The default `BaseDistribute` pipeline (`SPIRV/Passes.cpp:305-318`):

| line | pass | role |
|---|---|---|
| `306` | `addTileAndDistributeToWorkgroupsPasses` (→ `:123` `createTileAndDistributeToWorkgroupsUsingForallOpPass`) | create the **workgroup** `scf.forall` from the anchor op |
| `308` | `addBufferizePasses` (uses `gpuCopyFn`, `:140`) | one-shot bufferize |
| `311` | `SPIRVTileAndDistributePass` | tile to invocations |
| `312` | `MemrefCopyToLinalgPass` | convert `memref::CopyOp` → `linalg::GenericOp` |
| `313` | **`createGPUDistributeSharedMemoryCopyPass`** | distribute shared-mem copies to **threads** (linalg tiling, cyclic) |
| (post-bufferize, `addSPIRVBufferizePasses`) `:156` | **`createGPUDistributeScfForPass`** | distribute `scf.for` carrying `iree.gpu.distribute_dim` → `gpu::ThreadIdOp` |
| (module) `:653-654` | `createSPIRVLowerExecutableTargetPass` then **`createVerifyWorkgroupDistributionPass`** | the **Common** verifier, at the very end |

**The lowering-to-threads mechanism differs sharply between the two models:**

- *LLVMGPU*: a workgroup-mapped `scf.forall` is *left intact*; **non**-workgroup (thread/warp/lane)
  foralls are mapped to threads. `GPUDistributePass` (`GPU/GPUDistribute.cpp:96-111`) walks all foralls,
  and for any that do **not** carry a `WorkgroupMappingAttr` (`:97-100`) calls `mapNestedForallToThreadsImpl`
  (`:101-102`, `:51-54`) → upstream `mlir::transform::gpu::mapOneForallToThreadsImpl`, which rewrites the
  forall body in terms of `gpu::ThreadIdOp`. `GPUDistributeForallPass` (`GPU/GPUDistributeForall.cpp`)
  then resolves thread/warp-mapped foralls into an `scf.for` + `affine.delinearize_index` of a flat
  thread id (`:35-159`, forall→`scf.for` at `:133`, delinearize at `:147-148`), guarded by barriers
  (`:132`,`:134`). It **rejects** foralls with results ("Cannot distribute scf.forall op on tensors",
  `:57-60`) and non-normalized foralls (`:76-79`).
- *SPIR-V*: does **not** lower via `scf.forall` thread-mapping for the legacy path. `GPUDistributeScfForPass`
  (`GPU/GPUDistributeScfFor.cpp`) is a pattern on `scf::ForOp` that fires only when the loop carries the
  `iree.gpu.distribute_dim` marker (`:40-45`); it rewrites the loop's lower bound to `threadId*step+lb`
  (`:81-83`) and step to `count*step` (`:84-85`) using `gpu::ThreadIdOp` (`:67`) — a classic strided-thread
  loop distribution. Shared-memory copies are distributed by `GPUDistributeSharedMemoryCopyPass` via
  **linalg tiling + cyclic distribution** (see A.2), *not* by forall-wrapping.

> A.1 takeaway: there is **no single "GPU distribution pass"** — there are two families, and only the
> LLVMGPU family produces the `scf.forall`-of-subviews shape the NOTE cites as "the reference." The
> SPIR-V family (which Metal uses) distributes copies a *different* way (linalg tiling of
> copies-to-workgroup-memory).

### A.2 Shared-memory staging for strided copies — how the GPU expresses a 2-D-strided store

`GPUDistributeSharedMemoryCopyPass` (`GPU/GPUDistributeSharedMemoryCopy.cpp`) is the SPIR-V/Metal
shared-memory-copy distributor. Core mechanics:

- It collects `linalg::GenericOp`s that are copies **into workgroup memory** — selected by the
  `getCopyToWorkgroupMemoryMarker()` (the string `"copy_to_workgroup_memory"`,
  `Common/Utils/MarkerUtils.cpp:114-116`); the pass `walk`s and keeps only ops carrying that marker
  (`GPU/GPUDistributeSharedMemoryCopy.cpp:388-393`). **Critically, that marker is set *only* by the
  promotion machinery that copies *into* workgroup memory** — `Common/Utils/GPUUtils.cpp:272-273`
  (`createCopyToWorkgroupMem`: builds a `memref::CopyOp` to a workgroup-space dst and marks it),
  `SPIRV/Passes.cpp:106-108`, and `SPIRVTileAndPromote.cpp:316`. So this pass matches **copies whose
  destination is `#gpu.address_space<workgroup>` (shared memory), never copies to global output
  bindings.** The bug's stranded store-to-output-binding is unmarked and is therefore never touched.
- It then tiles each such copy so the iteration count equals the flat workgroup size
  (`getTileToDistributableSize`, `:134-157`) and distributes the resulting loops **cyclically** across
  threads via `DistributionMethod::CyclicNumProcsEqNumIters` (`:214-216`, `:235-269`), using a flat
  thread id (`createFlatId`, `:294-309`) delinearized over the loop ranges (`getIds`, `:191-219`).
- After distribution it vectorizes the copies to 128-bit `transfer_read/transfer_write`
  (`vectorizeCopyToWorkgroupMemoryOps`, `:273-291`; `copyVectorNumBits=128`, `:52`).

**How a 2-D-strided store is actually expressed — grounded from the `metal-spirv` reproduction.**
I ran the reduced reproducer against `metal-spirv` (exit 1; the offending IR fragment is in the error
context). The strided destination is expressed as a **`memref.subview` whose `static_strides` bake the
stride into the destination's memref type**, and the copy is a load/store loop over a dense shared-mem tile:

```
%41 = memref.subview %40[...]  static_strides = array<i64: 2, 1>  // OUTPUT binding, dim-0 strided -> strided<[?, 1]>
%43 = memref.subview %38[...]  static_strides = array<i64: 1, 2>  // input,          dim-1 strided -> strided<[?, 2]>
%44 = memref.alloc ... : memref<?x?xi8, #gpu.address_space<workgroup>>   // shared-mem scratch (fill result)
%47 = memref.alloc ... : memref<?x?xi8, #gpu.address_space<workgroup>>   // shared-mem staging buffer
// phase 1: input %38 (strided<[?,1]>) -> shared %47   (read-modify-write, preserves zeros)
// phase 2: shared %44 (strided<[?,1]>) -> shared %47 (strided<[?,2]>)   (dim-1 strided fill into scratch)
// phase 3: shared %47 -> OUTPUT %41 (strided<[?,1]>)                    <-- THE OFFENDING STORE
```
Each phase is a nest of `scf.for` loops doing `memref.subview` (unit `static_strides = [1,1]` on the
*inner* subview) + `memref.load` + `memref.store` between a shared-mem tile and the strided global subview.

The LLVMGPU analogue is even cleaner and is the best "reference shape": `distributeCopyToThreads`
(`GPU/GPUDistributeCopyUsingForall.cpp:47-112`) builds **unit-stride subviews of both operands**
(`strides(rank, getIndexAttr(1))`, `:106`) and runs a `memref::CopyOp` of those tiles inside a thread-mapped
forall (`:74-111`). The destination's stride is carried entirely by `copy.getTarget().getType()`; the loop
body sees only unit-stride tiles.

> A.2 takeaway: the GPU never "dilates a loop." It (i) bakes the stride into the **destination memref
> type** via a strided `memref.subview`, (ii) stages through a dense shared-memory tile, and (iii) emits a
> **unit-stride** load/store loop that scatters into the strided global view. The stride is a property of
> the *memory view*, not of the *loop index mapping*.

### A.3 Borrowability for a workgroup-level fix

**What transfers (cleanly) to a *workgroup*-mapped forall:**
- The **IR shape** `strided-memref.subview + unit-stride-inner-subview + copy-inside-forall` from
  `GPUDistributeCopyUsingForall.cpp:104-112`. Re-pointing the `mapping` from `gpu::GPUThreadMappingAttr`
  (`:69`) to `IREE::Codegen::WorkgroupMappingAttr` (the type the workgroup tiler already mints,
  `TileDispatchUsingForall.cpp:158-163`) yields exactly the construct a workgroup-level fix needs: each
  workgroup takes a unit-stride tile of its own strided output region and writes it. No new affine
  machinery is required — the stride is already expressible as `memref.subview ... static_strides`.
- The **correctness-by-disjointness** property the GPU relies on (unit-stride tiles of non-overlapping
  regions) is identical to the per-workgroup-corner scheme in NOTE §3.

**What does *not* transfer / is irrelevant to a workgroup fix:**
- **Shared-memory staging has no CPU analogue** and is *not required* for correctness. On CPU there is no
  `#gpu.address_space<workgroup>`; a workgroup forall that writes a unit-stride tile directly to the
  strided global subview is correct (each workgroup's tile is disjoint — NOTE §3). Staging is a GPU
  *coalescing/performance* optimization, not a correctness precondition.
- **The op-match.** `GPUDistributeCopyUsingForall` only fires on `memref::CopyOp` (`:138`), and the SPIR-V
  `GPUDistributeSharedMemoryCopy` only fires on `linalg::GenericOp` copies *into workgroup memory*
  (`:388-393`). The CPU bug produces bare `linalg.generic { yield %in }` copies to the **output binding**
  (`cpuCopyFn` → `createLinalgCopyOp`, per the main agent), which neither GPU pass would match. Borrowing
  therefore means borrowing the *generator logic*, not reusing the pass as-is.
- **The mapping level.** `GPUDistributeCopyUsingForall`'s single-thread fallback (`distributeCopyToSingleThread`,
  `:33-44`) and thread mapping are GPU-specific; a workgroup fix must produce a `WorkgroupMappingAttr`
  forall (which then satisfies `VerifyWorkgroupDistributionPass`, see B.5).

> A.3 takeaway: borrow `distributeCopyToThreads`'s *generator* (strided-subview + unit-stride-inner-tile +
> copy-in-loop) and re-target it to workgroup mapping. Do **not** borrow shared-memory staging (no CPU
> analogue) and do **not** expect to reuse either GPU pass unmodified (wrong op-match + wrong mapping
> level). This directly feeds Approach 1's tensor-level design: the tensor-level analogue is a strided
> `tensor.insert_slice` whose per-tile slice is unit-strided, fused into the workgroup forall.

### A.4 NOTE §4 coalescing + generality — answered with citations

**Coalescing / performance.** NOTE §4 asks whether a per-workgroup strided write is non-coalesced and
needs shared-memory staging. Because the bug **also fails on GPU**, I can answer empirically: **the GPU
pipeline does *not* hide the defect — it relocates it.** Reproducing `metal-spirv` (exit 1) shows the GPU
pipeline **does** stage through shared memory (`%44`, `%47` are `#gpu.address_space<workgroup>` allocs)
and **does** distribute the copies to threads (the `scf.for` load/store nests), exactly as
`GPUDistributeSharedMemoryCopy`'s cyclic distribution produces. But the **final storeback to the output
binding** (`memref.store %53, %52 : ... #hal.descriptor_type<storage_buffer>`) lands in those
thread-distributed `scf.for` loops, *outside* the workgroup `scf.forall`, and is rejected by
`VerifyWorkgroupDistributionPass` (`VerifyWorkgroupDistribution.cpp:72-74`, `:81-83`) — the **same** Common
verifier the CPU path fails, wired as a module pass *after* `createSPIRVLowerExecutableTargetPass`
(`SPIRV/Passes.cpp:653-654`). So NOTE §4's claim (lines 86-88: "the flagged op is a `memref.store` to the
output binding inside a *thread*-distributed `scf.for` … same root cause … different offending op") is
**confirmed accurate** — including that it is the Common `VerifyWorkgroupDistributionPass`, *not* the GPU's
own `GPUVerifyDistributionPass` (which would have phrased it "shared resources … lane or thread distributed
contexts," `GPUVerifyDistribution.cpp:125-127`, and would in fact *accept* a store inside a thread-mapped
forall by skipping its body, `:91-94`). The coalescing machinery runs but is moot: the store never reaches
a context the workgroup verifier accepts. `[Conclusion: for the proper fix, coalescing is a *secondary*
concern; the primary concern is getting the output writeback lexically inside the workgroup forall. A
direct strided write is acceptable for correctness on all targets; GPU shared-mem staging can be layered
on later for performance if benchmarks demand it.]`

**Generality — what stride patterns does the GPU copy-distribution accept?**
`distributeCopyToThreads` (`GPUDistributeCopyUsingForall.cpp:47-112`) is **stride-agnostic**: it takes
unit-stride subviews of *whatever layout* the source/target memrefs already carry (`:104-111`), so it
accepts **arbitrary per-dimension strides** baked into the memref type (e.g. `strided<[?,1]>` *and*
`strided<[?,2]>` simultaneously — exactly the axis-mismatched pair in this bug). The tile sizes come from
`getCopyTileSizes` (called at `:147`) which keys off the workgroup size, not the strides. So the borrowed
shape generalizes to `[a,b]` strides and higher ranks *for free*, because the loop body never inspects the
stride — the view does. (The `linalg.generic`/`memref.store` scatter it lowers to handles arbitrary strides
element-wise.) `[INFERENCE: this is also why the GPU path can even *represent* the doubly-strided copy — it
just can't get the resulting store into a workgroup forall, which is a fusion/tiler problem, not a
stride-expressiveness problem.]`

---

## Part B — Survey: is anything similar being done? how are similar distribution ops executed?

### B.5 Every "distribute X into forall" pass in `Codegen/`, with mapping-level classification

I located every `scf::ForallOp::create` / forall-rewriting pass and every `gpu::ThreadIdOp` /
thread-mapping site under `Codegen/`. "WG" = workgroup-level (the level `VerifyWorkgroupDistributionPass`
accepts); "THRD/WARP/LANE" = GPU sub-workgroup levels.

| # | Pass | File:line (core) | Mapping level | What it distributes | Op matched / created |
|---|---|---|---|---|---|
| 1 | **`TileAndDistributeToWorkgroupsUsingForallOpPass`** | `Common/TileDispatchUsingForall.cpp:228` (`runOnOperation`), anchor pick `:68-76`, forall built `:353-355` (`LoopType::ForallOp` `:301`) | **WG** | the workgroup-*anchor* compute op (last compute op w/ a workgroup tiling level, `:64-76`) into the workgroup `scf.forall` | tiles a `TilingInterface` op; **does not** fuse strided writeback (consumer fusion delegated to `fuseConsumersIntoForall`, row 2) |
| 2 | *(helper, not a pass)* **`fuseConsumersIntoForall`** | `Common/TileAndFuseUtils.cpp:112-247`, filter `:141` | WG (fuses into #1's forall) | consumers of the tiled anchor — **only `tensor::ParallelInsertSliceOp`** (`:141-144`) | the strided store is **not** a fusible consumer here → stranded (root cause) |
| 3 | **`LLVMGPUTileAndDistributePass`** | `LLVMGPU/LLVMGPUTileAndDistribute.cpp:34-65` | WG (re-tiles reduction loops) | extra reduction-dim tiling at workgroup level (workgroup tiling "is done at the flow level", `:31-33`) | `PartitionableLoopsInterface` ops |
| 4 | **`GPUDistributeCopyUsingForallPass`** | `Common/GPU/GPUDistributeCopyUsingForall.cpp:117-151`, match `:138`, gen `:47-112` | **THRD** | each `memref::CopyOp` → a thread-mapped forall of unit-stride subviews | `memref::CopyOp` (only) |
| 5 | **`GPUDistributePass`** | `Common/GPU/GPUDistribute.cpp:73-117` | THRD (resolves non-WG foralls) | non-WG foralls → `gpu::ThreadIdOp` via `mapOneForallToThreadsImpl` (`:51-54`, `:101-102`); skips WG-mapped (`:97-100`) | `scf::ForallOp` without `WorkgroupMappingAttr` |
| 6 | **`GPUDistributeForallPass`** | `Common/GPU/GPUDistributeForall.cpp:29-32`, `:161-223`, resolve `:35-159` | THRD / WARP | thread/warp-mapped foralls → `scf.for` + delinearized flat thread id (`:133`, `:147`); rejects tensor-result foralls (`:57-60`) | `scf::ForallOp` w/ `GPUThreadMappingAttr`/`GPUWarpMappingAttr` (`:47-55`) |
| 7 | **`GPUDistributeScfForPass`** | `Common/GPU/GPUDistributeScfFor.cpp:98-110`, pattern `:32-96` | THRD (SPIR-V model) | `scf::ForOp` carrying `iree.gpu.distribute_dim` marker → strided thread loop (`lb=threadId*step+lb`, `:81-83`) | `scf::ForOp` w/ `getGPUDistributeAttrName()` (`:40-45`) |
| 8 | **`GPUDistributeSharedMemoryCopyPass`** | `Common/GPU/GPUDistributeSharedMemoryCopy.cpp:379-466`, walk `:388-393`, tile `:57-106`, cyclic dist `:235-269` | THRD (SPIR-V model) | `linalg::GenericOp` copies **into workgroup memory** → cyclic thread distribution + 128-bit vectorization (`:273-291`) | `linalg::GenericOp` w/ `getCopyToWorkgroupMemoryMarker()` (`:329`) |
| 9 | **`GPUGreedilyDistributeToThreadsPass`** | `Common/GPU/GPUGreedilyDistributeToThreads.cpp:29-33`, `:157-164`, tile `:43-110` | THRD | any `TilingInterface` op not already in a thread/warp/lane forall → tiled to threads via `iree_gpu.derived_thread_config` (`:36-42`); skips ops nested in mapped foralls (`:112-155`) | `TilingInterface` ops |
| 10 | **`CombineLayoutTransformationPass`** (+ `GPUCombineLayoutTransformation`) | `Common/CombineLayoutTransformation.cpp:282` (forall create), mapping `:948-953` | **WG** (or thread) | relayout (pad/transpose) fused into a workgroup forall whose terminator is a `parallel_insert_slice` (`CombineLayoutTransformation.h:56-57`); **rejects non-unit strides** (`:220-223`) | `iree_codegen.apply_layout_transformation` / `tensor.extract_slice` (unit-stride only) |
| 11 | **`ReconcileTranslationInfoPass`** | `Common/ReconcileTranslationInfo.cpp:231` (forall create), mapping verify `:56-73`, collapse `:170-242` | **WG** | merges/collapses workgroup-mapped foralls to reconcile the dispatch's translation info (`:386-393`) | `scf::ForallOp` w/ `WorkgroupMappingAttr` (`:389-390`) |
| 12 | **`ConvertWorkgroupForallToPCFPass`** | `Common/ConvertWorkgroupForallToPCF.cpp:58-61` | WG (→ PCF workgroup scope) | WG-mapped forall → `pcf.loop` w/ workgroup scope, linearizing ids (`Passes.td:164-171`) | `scf::ForallOp` w/ `WorkgroupMappingAttr` |
| 13 | **`GPUPackPartialReductionsPass`** | `Common/GPU/GPUPackPartialReductions.cpp` | **WG** | packs partial-reduction dims into a workgroup forall | reduction producers |
| 14 | *(lane level)* **`mapLaneForalls`** (called by #5 `:80`, #6 `:166`) | `Dialect/GPU/Transforms/Transforms.cpp:588-592` (warp→thread forall), `:825`, `:1071`, `:1477` | LANE / WARP / THRD | lane-mapped foralls → lane ops; warp forall → thread forall | `scf::ForallOp` w/ `IREE::GPU::LaneIdAttr`/`GPUWarpMappingAttr` |
| 15 | *(transform-dialect)* **`IREE::GPU` apply-transforms** | `Dialect/GPU/IR/IREEGPUAttrs.cpp:3230` (forall create), `Dialect/GPU/Transforms/Transforms.cpp` | THRD/WARP/LANE/WG (configurable) | transform-dialect tiling/distribution primitives | transform-target ops |
| 16 | **PCF distribute** (`ConvertForallToPCF`, `FoldForallIntoPCFLoop`, `FuseConsumers`, `LowerStructuralPCF`) | `Dialect/PCF/Transforms/ConvertForallToPCF.cpp:459`, `FoldForallIntoPCFLoop.cpp:394`,`:441`, `FuseConsumers.cpp:1260`, `LowerStructuralPCF.cpp:167` | WG/THRD (PCF scopes) | producer-consumer-framework forall construction/hoisting | `scf::ForallOp` ↔ `pcf.generic`/`pcf.loop` |

**Classification relevant to the bug.** Only the **WG-level** passes (#1, #3, #10, #11, #12, #13) produce
the construct `VerifyWorkgroupDistributionPass` accepts (a write lexically nested in a `WorkgroupMappingAttr`
forall — `VerifyWorkgroupDistribution.cpp:48-56`). The thread/warp/lane passes (#4-#9, #14) are GPU-specific
and, as A.4 showed, the GPU's own thread-level distribution does **not** satisfy the Common workgroup
verifier (a thread loop is *not* a workgroup forall). So "is anything similar being done" for *workgroup*-level
strided writes: **only #1's anchor tiling + #2's `ParallelInsertSliceOp`-only consumer fusion**, and #2 is
the exact filter that strands the store. No pass distributes a *strided writeback* into a workgroup forall.

**Two verifiers, not one — and the bug hits the Common one on both backends:**
- `VerifyWorkgroupDistributionPass` (`Common/VerifyWorkgroupDistribution.cpp:48-83`): any write to a global
  memref (`hasGlobalMemoryAddressSpace`, `:68`) outside a WG-mapped forall is an error (`:72-74`). **This is
  the one both llvm-cpu and metal-spirv fail.** It is a *module* pass run at the very end
  (`SPIRV/Passes.cpp:653-654`, `LLVMCPU/Passes.cpp:705-706`, `LLVMGPU/Passes.cpp:1173-1174`).
- `GPUVerifyDistributionPass` (`Common/GPU/GPUVerifyDistribution.cpp:31-137`): stricter, GPU-only, requires
  writes in *thread/lane* contexts (`:91-94`); skips thread/lane-mapped forall bodies, *advances into*
  warp-mapped bodies (`:96-98`); allows DMA copies (`:120-123`). Wired only in the LLVMGPU pipeline
  (`LLVMGPU/Passes.cpp:596`). The bug does **not** reach this pass (it fails the Common verifier first).

### B.6 Existing dilation / stride-distribution primitive — answer: **NO**

NOTE §4's open question ("Dilation expressiveness: is there an existing affine/dilation primitive for
'each workgroup writes every-k-th element along a dimension'?"): **No.** Evidence — every site I found that
touches strides in the *distribution/fusion* path **bails on non-unit strides**:

- `Common/CombineLayoutTransformation.cpp:220-223` — `if (!areAllConstantIntValue(extractSliceOp.getMixedStrides(), 1)) return "non-unit strides are not supported";`
- `Common/ReshapePatterns.cpp:415-416` — `if (!areAllConstantIntValue(storeOp.getMixedStrides(), 1)) return "found a non-1 stride";`
- `Common/ReshapePatterns.cpp:622-623` — `if (!all_of(storeOp.getMixedStrides(), isOneInteger)) return "expected unit strides";`
- `Common/IREECodegenCanonicalizer.cpp:31-32` — rejects non-unit-offset/non-unit-stride subviews.
- `Common/TensorDynamicDimAnalysis.cpp:110-113` — aborts unless all strides are 1.

The only `strides`/`dilations` attributes present in the tree are **convolution window** attributes
(`linalg.conv_2d_*.{strides,dilations}`, `iree_linalg_ext.im2col {strides, dilations, kernel_size}`)
— e.g. `GPU/test/gpu_create_fast_slow_path.mlir:38` (`strides = dense<2>`), `generic_vectorization_*`.
These are consumed by conv-specific tiling (`GPUTileAndConvertConvToMatmul`, `GPUCreateFastSlowPath`),
*not* by any general "distribute a strided output write" primitive. `GPUDistributeCopyUsingForall` is
stride-**agnostic** (A.4) but it is a copy-distributor, not a writeback-dilation primitive, and it matches
`memref::CopyOp` only.

> B.6 takeaway: **there is no existing primitive for per-workgroup strided/dilated writes.** The capability
> to add — a fusible strided writeback that places a unit-stride tile of a strided output region inside the
> workgroup forall — does not exist anywhere in `Codegen/`. This is the core thing Approach 1 must build.
> (A.2/A.3 show the *shape* it should produce already exists in the GPU copy-distributor.)

### B.7 How the single-stride control executes today (classification only)

The passing single-stride control (`m[:,0::2]`-style) compiles. From `control_dump.mlir` at the
`@main$async_dispatch_0_transpose_DxD_i1` function (`control_dump.mlir:25814-25889`):

- The output binding is subviewed once into a **dim-0-strided** view (`strided<[?, 1]>`):
  `control_dump.mlir:25851` `%subview = memref.subview %24[0,0][%22,%25][1,1] ... strided<[?, 1]>`.
- The fill and the store execute **inside the workgroup `scf.forall`** (`:25852`, mapping
  `[workgroup y, workgroup x]` at `:25883`), writing **directly to the output-binding subview in place**:
  - vectorized fill into the output tile: `:25859` `vector.transfer_write %cst, %arg5[%arg2,%arg4] : vector<1x4xi8>, memref<...strided<[?,1]>... #hal.descriptor_type<storage_buffer>>`
  - tail fill into the output tile: `:25865` `linalg.generic ... outs(%subview_2 : ...storage_buffer) { yield %c1_i8 }`
  - an in-place copy within the same tile (`ins==outs` of the output tile subview): `:25879-25882`.
- The `%arg3`/`%arg5` iter_args of the inner `scf.for` are **the output tile subview itself** (`:25856`,
  `iter_args(%arg3 = %subview_0)`), i.e. the dispatch's output store **aliases the output binding in place
  inside the workgroup forall**.

**Classification / answer to "how are similar distribution ops executed":** in the single-stride control,
the dispatch's output write **is** the workgroup forall's body — the fill writes straight into a unit-stride
tile of the (already-dim-0-strided) output subview, so the store is lexically inside the WG forall and
satisfies `VerifyWorkgroupDistribution.cpp:48-56`. This confirms NOTE §4 line 34 ("single-strided stores
already alias in-place inside the loop"). *Why it works for one axis but not two:* with a single shared
axis, the fill region and the output region have the **same** stride pattern, so a unit-stride tile of the
fill *is* a unit-stride tile of the output → in-place aliasing is legal. With two axis-disjoint strides
(the bug), the fill region (dim-1 strided) and the output store (dim-0 strided) have **incompatible** layouts,
bufferization cannot alias them in place, and the store is emitted outside the forall as a gather/scatter.
(The *generalization* of this one-axis precedent to two axes is Approach 1's design deliverable; not
duplicated here.)

---

## 5. What this approach contributes

**To the proper fix (Approach 1, tensor-level fusion):**
1. A **concrete, source-grounded IR shape to target** — `distributeCopyToThreads`'s generator
   (`GPUDistributeCopyUsingForall.cpp:104-112`): *strided destination view + unit-stride per-tile inner
   subview + copy-in-loop*. The tensor-level analogue is a strided `tensor.insert_slice` whose per-workgroup
   tile is unit-strided, fused into the WG forall. No new affine/dilation primitive is *required* to express
   the stride — `memref/tensor` subview strides already carry it (A.2, A.4).
2. **Proof that the stride-expressiveness gap is closed by the view, not the loop** (A.4 generality) — so
   the fix generalizes to arbitrary `[a,b]` strides and higher ranks without new loop-dilation machinery.
3. **Confirmation the capability genuinely does not exist** (B.6) — Approach 1 is building net-new
   fusion, not reusing a hidden primitive.

**To the stopgap (Approach 2, post-bufferization memref-level distribution):**
1. `GPUDistributeCopyUsingForall` is the **closest existing template** for a CPU stopgap (NOTE §4 Idea 2),
   but it must be retargeted: match the stranded `linalg.generic { yield %in }` copy (not `memref::CopyOp`),
   emit a `WorkgroupMappingAttr` forall (not thread mapping), and skip shared-memory staging (no CPU analogue).
2. **A warning to carry into the stopgap:** A.4 shows the GPU's own copy-distribution does *not* by itself
   satisfy the Common workgroup verifier — a stopgap that distributes into a *thread* loop (mirroring the GPU)
   will still fail `VerifyWorkgroupDistribution.cpp:72-74`. The forall **must** be workgroup-mapped.

**Headline answers to the user's two questions:**
- *"Is anything similar being done in IREE?"* — For *copy distribution* (to threads/shared-mem): yes,
  extensively (B.5 #4-#9). For *strided-writeback distribution into a workgroup forall*: **no** — the only
  relevant fusion (`fuseConsumersIntoForall`, B.5 #2) filters to `ParallelInsertSliceOp` and every
  stride-aware site rejects non-unit strides (B.6).
- *"How are similar distribution ops executed?"* — Copies are executed as **unit-stride load/store loops
  over a strided destination view** (stride in the memref type, not the loop), distributed to threads either
  by forall-thread-mapping (LLVMGPU) or by linalg-tiling+cyclic-distribution / `scf.for`+`distribute_dim`
  (SPIR-V). Workgroup-mapped writes are executed as the *body* of the workgroup forall, aliasing the output
  in place (B.7). The defect is precisely that the doubly-strided writeback takes neither path.
