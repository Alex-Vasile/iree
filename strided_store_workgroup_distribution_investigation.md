# Investigation note — distribute the strided output store into the workgroup loop

Status: **idea / basis for implementation investigation.** Not implemented. This note
captures (1) what the idea is, (2) where it comes from, (3) a concrete example, then
points an implementer at the relevant code and the open questions to resolve.

All paths below are relative to `compiler/src/iree/compiler/` unless noted.
The full source-grounded bug investigation (per-pass IR dumps, lowering traces, an
independent staff review, and the reduced reproducer) lives in the sibling editor repo:
`../editor/rcd_lowpass_llvm_cpu_repro/` (absolute: `/Users/alex/Developer/editor/rcd_lowpass_llvm_cpu_repro/`).
Start there with `OVERVIEW.md`.

---

## 1. What the idea is

**Teach the workgroup tiler to fuse/distribute a *strided* output store into the
workgroup `scf.forall`, so each workgroup writes its own non-overlapping strided
sub-pattern of the output — instead of leaving the strided store stranded outside the
loop as a sequential copy.**

Today `TileAndDistributeToWorkgroupsUsingForallOp` tiles only the tiling-config-carrying
"anchor" op (typically a fill / compute) and wraps it in the workgroup `scf.forall`. A
strided write-back (the dispatch's output store) is **not** fused into that loop, because
`fuseConsumersIntoForall` only collects `tensor::ParallelInsertSliceOp` consumers
(`Codegen/Common/TileAndFuseUtils.cpp:141-144`). The stranded store is then materialized
by comprehensive bufferization as a bare sequential copy at dispatch-function scope, which
`VerifyWorkgroupDistributionPass` rejects.

The proposal: give the tiler a code path that, for a strided `tensor.insert_slice` /
`store_to_buffer` whose stride pattern is a clean per-dimension dilation (e.g. `[1,2]` or
`[2,1]` or `[2,2]`), fuses it into the workgroup loop with the appropriate affine dilation
so that **each workgroup writes the strided slice of its own tile** — a 2-D-strided
analogue of how single-strided stores already alias in-place inside the loop.

This is the "Idea 1" proper fix from the investigation (distinct from "Idea 3" = avoid
creating the transposed dispatch at flow formation, and from "Idea 2" = patch the stray
copies post-bufferization).

---

## 2. Where it comes from

### The bug
`TestRCDLowPass::test_can_compile[llvm-cpu]` in the editor repo fails to compile
`RCDLowPassModule` (`../editor/src/python/editor/compute/rcd_demosaic.py`). The module
builds boolean CFA masks via doubly-strided, out-of-place assignment on **dynamic**
`[?,?]` tensors:

```python
r_mask = torch.zeros((H, W), dtype=torch.bool); r_mask[0::2, 0::2] = True
b_mask = torch.zeros((H, W), dtype=torch.bool); b_mask[1::2, 1::2] = True
```

### The error
```
'linalg.generic' op write affecting operations on global resources are restricted
   to workgroup distributed contexts.
'func.func' op failed on workgroup distribution verification
```

### The confirmed root-cause chain (each link source-cited)
1. **Flow dispatch formation** fuses the two axis-disjoint scatters of `m[0::2,0::2]=True`
   into one **transposed** read-modify-write dispatch: the fill region ends up strided on
   dim 1 (`insert_slice [1,2]`) while the output store is strided on dim 0
   (`store ... strides=[2,1]`). Dispatch `@main$async_dispatch_0_transpose_DxD_i1`.
2. **`TileAndDistributeToWorkgroupsUsingForallOpPass`** (`Common/TileDispatchUsingForall.cpp:228`
   `runOnOperation`) tiles **only** the fill anchor into a workgroup `scf.forall`. The
   strided output store is not a `ParallelInsertSliceOp`, so `fuseConsumersIntoForall`
   (`Common/TileAndFuseUtils.cpp:141-144`) skips it → it stays **outside** the forall.
3. **`IREEComprehensiveBufferizePass`** (`Common/IREEComprehensiveBufferize.cpp:263`),
   with the llvm-cpu copy fn `cpuCopyFn` → `createLinalgCopyOp`
   (`Common/CPU/Passes.cpp:40-44`, `Common/Utils/Utils.cpp:2287`), cannot alias the
   dim-1-strided fill tensor to the dim-0-strided output (axis mismatch), so it emits bare
   `linalg.generic` gather/scatter copies (+ a scratch alloca) **at dispatch-function
   scope, outside the forall**.
4. **`VerifyWorkgroupDistributionPass`** (`Common/VerifyWorkgroupDistribution.cpp:29-82`)
   enforces: *any write to a global memref (`#hal.descriptor_type<storage_buffer>`) must be
   lexically inside a workgroup-mapped `scf.forall`.* The stray copies violate this → error
   at `:72`.

### Why this is the right level to fix (it is backend-general)
The failure is **not** llvm-cpu-specific. The same dispatch fails the **same** verifier on
**`metal-spirv`** too (verified 2026-06-27: exit 1, both error substrings, identical
dispatch name; `vulkan-spirv` shares the SPIR-V pipeline and behaves the same). On GPU the
flagged op is a `memref.store` to the output binding inside a *thread*-distributed
`scf.for` (not a workgroup `scf.forall`) instead of a bare `linalg.generic` — same root
cause (the output store never enters a workgroup-mapped forall), different offending op.
Because the defect is born at flow formation / codegen-common and the verifier is in
`Codegen/Common/`, a fix at the tiler/distribution layer fixes all backends at once.

This directly motivates the idea: the operation **is** parallelizable and race-free (see
§3); what's missing is the codegen code path that actually generates that parallelization.

---

## 3. Example — and why it is parallelizable (the key insight)

### The operation, in memory
`m` is stored **row-major**: address = row × W + col. For a 4×4 `m` (W=4):

```
        col0 col1 col2 col3
row0      0    1    2    3
row1      4    5    6    7
row2      8    9   10   11
row3     12   13   14   15
```

`m[0::2, 0::2] = True` writes positions (0,0),(0,2),(2,0),(2,2) → **addresses 0, 2, 8, 10**.
The gaps are `+2, +6, +2` — **no single stride**: two independent skip rules (columns by 2,
rows by 2×W). That is "strided in two dimensions."

### The per-workgroup distribution (what the idea would generate)
Tile the dense sub-grid of Trues into 2×2 workgroup tiles. Each workgroup writes its 4
Trues into the corners of its own **disjoint** 4×4 output block:

```
workgroup computed (dense 2x2):      lands in output m (its own 4x4 block):
 T T                                  T . T .
 T T                                  . . . .
                                      T . T .
                                      . . . .

workgroup (0,0) -> corners of output rows[0:4], cols[0:4]
workgroup (0,1) -> corners of output rows[0:4], cols[4:8]
workgroup (1,0) -> corners of output rows[4:8], cols[0:4]
...
```

No two workgroups touch the same output cell → **no race, fully parallelizable, correct.**
The compute-tile (i,j) maps to output addresses via a 2-D dilation `(2i, 2j)`. This is the
affine relationship the tiler would need to express.

### Why IREE currently emits something illegal instead
The tiler tiles the fill into a workgroup loop, but the strided **store** is left outside it
(not a `ParallelInsertSliceOp`, so not fused). Bufferization then turns that stranded store
into a **plain sequential copy** at function scope:

```
what the idea would generate:               what IREE emits today:
workgroup loop:                             workgroup loop:
   fill  ──┐                                  fill (into a temp scratch)
   store strided corners of my tile         ─────────────────────────────
                                             sequential copy: temp -> output
                                             (OUTSIDE the workgroup loop)  <-- illegal
```

`VerifyWorkgroupDistributionPass` is a **structural** guard (it checks lexical nesting in a
workgroup forall; it does *not* analyze address overlap), so the stray copy is rejected
even though the per-workgroup-corner scheme in the left column would be safe. The bug is a
**missing codegen capability**, not a correctness impossibility.

---

## 4. Implementation — approaches to investigate

This note is intentionally a starting point, not a design. Things to evaluate:

1. **Fuse at the tensor level (extend the tiler).** Widen `fuseConsumersIntoForall`
   (`Common/TileAndFuseUtils.cpp:112-202`) to also accept a strided `tensor.insert_slice`
   (and/or the `store_to_buffer` that `iree-codegen-bufferize-dispatch-tensor-load-store`
   produces) as a fusible consumer, generating the per-workgroup strided write with the
   correct dilation in the `scf.forall` body. *Question:* can the existing `scf.forall`
   `shared_outs` / `in_parallel` machinery represent a strided write-back, or does it
   require a new insert form?
2. **Fuse at the memref level (post-bufferization).** Add a CPU analogue of
   `GPUDistributeCopyUsingForallPass` (`Common/GPU/GPUDistributeCopyUsingForall.cpp:117-151`)
   that distributes the stranded `linalg.generic` copies into a workgroup forall. Cheapst
   correctness patch; downside: leaves the redundant gather/scatter + scratch in place and
   masks the flow/tile defect. (This is "Idea 2" — useful as a stopgap, not the proper fix.)
3. **Reuse GPU's distribution strategy.** The GPU path *does* distribute the copies (to
   threads, via shared-memory staging) — study how it expresses the 2-D-strided copy and
   whether a workgroup-level version can borrow that IR shape.

### Open questions to resolve before implementing
- **Dilation expressiveness:** is there an existing affine/dilation primitive for "each
  workgroup writes every-k-th element along a dimension" in either CPU or GPU codegen? If
  yes, reuse; if no, that's the core thing to add.
- **Zero-preservation / read-modify-write:** the dispatch is read-modify-write (it loads the
  whole input to preserve the non-scattered zeros). If the store is distributed
  per-workgroup, does the input load also need distributing, or can the output be assumed
  pre-zeroed / handled by a separate init? (Cf. the `m[0::2,0::2]=True` semantics: untouched
  cells keep their original value.)
- **Generality:** does the fix need to handle arbitrary per-dimension strides (`[a,b]`) and
  higher ranks, or is `[2,2]`-style uniform doubling the common case worth handling first?
- **Performance / coalescing:** a per-workgroup strided write is non-coalesced. Does it need
  shared-memory staging (as the GPU path does) to be efficient, or is direct strided
  storage acceptable on the targets of interest?
- **Interaction with the verifier:** confirm that once the store is lexically inside the
   workgroup forall, `VerifyWorkgroupDistributionPass` passes (it should — that's its only
   condition), and that no *other* invariant (e.g. shared_outs aliasing) is violated.

---

## 5. Key files (verified citations)

| Role | Location |
|---|---|
| Workgroup tiler (tiles only the anchor; strands the store) | `Common/TileDispatchUsingForall.cpp:228` (`runOnOperation`) |
| Consumer-fusion filter (only `ParallelInsertSliceOp` — why the store is skipped) | `Common/TileAndFuseUtils.cpp:141-144` (inside `fuseConsumersIntoForall`, `:112-202`) |
| One-shot bufferize driver | `Common/IREEComprehensiveBufferize.cpp:263` (`runOnOperation`) |
| llvm-cpu copy fn (the bare `linalg.generic`) | `Common/CPU/Passes.cpp:40-44` (`cpuCopyFn`), `:46-50` (`addCPUBufferizePasses`), `:21-38` (`cpuAllocationFn`) |
| Copy-op builder | `Common/Utils/Utils.cpp:2287` (`createLinalgCopyOp`) |
| GPU copy-distribution (reference / to borrow from) | `Common/GPU/GPUDistributeCopyUsingForall.cpp:117-151` |
| The structural verifier (must not be relaxed) | `Common/VerifyWorkgroupDistribution.cpp:29-82` |
| llvm-cpu pipeline wiring | `Codegen/LLVMCPU/Passes.cpp` (`DoubleTilingExpert`, `addCPUBufferizePasses`, verifier) |

---

## 6. Reproduce

Use the editor repo's venv binary by absolute path:
```bash
/Users/alex/Developer/venv_iree/bin/iree-compile --iree-hal-target-backends=llvm-cpu \
  ../editor/rcd_lowpass_llvm_cpu_repro/reduced_reproducer.mlir -o /dev/null
# expect: exit 1, both error substrings above

# Same failure on GPU/SPIR-V (confirms backend-general):
/Users/alex/Developer/venv_iree/bin/iree-compile --iree-hal-target-backends=metal-spirv \
  ../editor/rcd_lowpass_llvm_cpu_repro/reduced_reproducer.mlir -o /dev/null
# expect: exit 1, same substrings; flagged op is a memref.store in a thread scf.for

# Single-strided control (compiles fine — fill and store share one axis):
/Users/alex/Developer/venv_iree/bin/iree-compile --iree-hal-target-backends=llvm-cpu \
  ../editor/rcd_lowpass_llvm_cpu_repro/control_single_stride.mlir -o /dev/null
# expect: exit 0
```

Per-pass IR dump (to watch the store get stranded and then materialized):
```bash
/Users/alex/Developer/venv_iree/bin/iree-compile \
  --mlir-disable-threading --mlir-print-ir-before-all --mlir-print-ir-after-all \
  --mlir-print-ir-after-change --mlir-elide-elementsattrs-if-larger=8 \
  --iree-hal-target-backends=llvm-cpu \
  ../editor/rcd_lowpass_llvm_cpu_repro/reduced_reproducer.mlir -o /dev/null 2> dump.mlir
# 34774-line dump already captured at ../editor/rcd_lowpass_llvm_cpu_repro/dump.mlir
```

## 7. Provenance / prior art in the editor repo

- `../editor/rcd_lowpass_llvm_cpu_repro/OVERVIEW.md` — authoritative synthesis (start here)
- `../editor/rcd_lowpass_llvm_cpu_repro/expert_review.md` — independent staff review; this
  note's "Idea 1" is the expert's long-term single-dispatch fix recommendation
- `../editor/rcd_lowpass_llvm_cpu_repro/findings_dump.md` / `findings_lowering.md` —
  per-pass IR + source-level lowering traces
- `../editor/rcd_lowpass_llvm_cpu_repro/reduced_reproducer.mlir` — 23-line minimal repro
