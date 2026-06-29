---
hide:
  - tags
tags:
  - compiler
  - llvm-cpu
  - flow
---

# Canonicalizing doubly-strided writebacks (llvm-cpu)

*Status: implemented (single-file change in `compiler/src/iree/compiler/Dialect/Flow/Transforms/Canonicalize.cpp`).*

## TL;DR

A dynamic, doubly-strided out-of-place write such as PyTorch's
`m[0::2, 0::2] = value` failed to compile on `llvm-cpu` with
`VerifyWorkgroupDistribution` rejecting bare `linalg.generic` copies to
`#hal.descriptor_type<storage_buffer>` memrefs at dispatch scope. The defect is
born at flow dispatch formation, where the two axis-disjoint `slice_scatter`s
become a single **transposed** read-modify-write dispatch (a dim-1-strided
`tensor.insert_slice` feeding a dim-0-strided `dispatch.tensor.store`). This
document records the fix that was chosen, why it was chosen over the
alternatives, and the dead ends hit along the way.

The fix is a new tensor canonicalization, `FoldNestedInsertSlice`, that
collapses such a nested write through a value-equivalent intermediate into a
single in-place multi-strided write. For the reproducer this turns the
transposed `insert[1,2] + store[2,1]` chain into a single
`dispatch.tensor.store ... strides=[2, 2]`, which workgroup tiling distributes
normally. It is a pure tensor identity, touches no verifier and no shared
fusion heuristic, and the full compiler test suite (1153 lit/ctest cases) passes
with no regressions.

## Background: the symptom and reproducer

The trigger is a boolean CFA mask built with a doubly-strided out-of-place
assignment and **dynamic** `[?, ?]` shapes:

```python
m = torch.zeros((H, W), dtype=torch.bool)
m[0::2, 0::2] = True
```

which lowers (torch-mlir → IREE) to two sequential, axis-disjoint
`aten.slice_scatter`s. Minimal reproducer (23 lines, torch dialect) and the
expected failure oracles are checked into the source investigation under
`rcd_lowpass_llvm_cpu_repro/` (external to this repo). The two oracle error
substrings:

```text
'linalg.generic' op write affecting operations on global resources are restricted to workgroup distributed contexts.
'func.func' op failed on workgroup distribution verification
```

The single-stride controls (`m[:, 0::2]`, etc.) compile cleanly; only the
**doubly**-strided, dynamic, out-of-place writeback fails.

## Confirmed root cause (layered defect)

Each link below was verified against source and per-pass IR dumps.

1. **Origin — flow dispatch formation.** The two axis-disjoint `slice_scatter`s
   are fused into one dispatch whose body is a transposed read-modify-write: a
   `linalg.fill` of a `[H/2, W/2]` region, a dim-1-strided `tensor.insert_slice
   ... [1, 2]` (into an intermediate), and a dim-0-strided
   `iree_tensor_ext.dispatch.tensor.store ... strides=[2, 1]` (the output
   writeback). The fill region and the output store are strided on **different
   axes** — this is the divergence.
2. **Propagation — workgroup tiling.**
   `TileAndDistributeToWorkgroupsUsingForallOpPass` tiles only the
   `lowering_config`-carrying fill anchor and leaves the strided output store at
   dispatch-function scope, outside the only workgroup `scf.forall`. The
   writeback is not a fusion candidate for `fuseConsumersIntoForall` (it is a
   plain `tensor.insert_slice` + `store_to_buffer`, not a
   `tensor.parallel_insert_slice`, and `store_to_buffer` is not a compute op).
3. **Materialization — comprehensive bufferization.** Because the fill region
   (dim-1 strided) and the output store (dim-0 strided) have incompatible
   layouts, one-shot bufferization cannot alias the fill tensor onto the output
   buffer. It allocates scratch and emits bare `linalg.generic` gather/scatter
   copies (via `cpuCopyFn` → `createLinalgCopyOp`) at dispatch-function scope.
4. **Detection — the verifier.** `VerifyWorkgroupDistributionPass` correctly
   rejects any write to a global (`storage_buffer`) memref outside a
   workgroup-mapped `scf.forall`.

### The verifier is correct and load-bearing

`Codegen/Common/VerifyWorkgroupDistribution.cpp` enforces: *every write to a HAL
`storage_buffer` must be lexically nested inside a workgroup-mapped
`scf.forall`.* This invariant is required for GPU correctness too, so it must
not be relaxed. The defect must be fixed upstream of the verifier, not in it.

## Goals and non-goals

### Goals

- Make the doubly-strided dynamic out-of-place writeback compile on `llvm-cpu`
  (and on any backend) with correct results.
- Fix the defect at its origin (flow formation) rather than papering over it
  post-bufferization.
- Keep the change small, contained, and non-regressing: do not weaken the
  verifier and do not perturb the shared flow-fusion heuristics.

### Non-goals

- Solving the general "distribute an arbitrary transposed store" codegen
  problem (a much larger, riskier change — see "Idea 1" below).
- Fixing the two **separate** secondary bugs observed in the failing dispatch
  (the `2^52` scratch alloca; `cpuAllocationFn` preserving the descriptor memory
  space on scratch allocs). Those are independent and out of scope here.

## Options considered

The investigation produced five candidate fix sites. They are summarized with
the verdict reached during this work.

| Idea | Site | Verdict |
|---|---|---|
| 1 | Teach workgroup tiling to distribute the transposed output store | **Rejected as the primary fix.** Correct but large and high-risk: it touches the core workgroup-distribution strategy used by every dispatch. Useful long-term if a single-dispatch solution is ever required. |
| 2 | Wrap the bare post-bufferization copies into a workgroup `scf.forall` | **Rejected.** Cheapest correctness patch but mis-layered: leaves the redundant gather + scratch chain intact (including the `2^52` alloca), and there is no existing CPU pass to reuse. Masks the real defect. |
| 3 | Prevent flow formation from producing the transposed dispatch (don't fuse axis-disjoint scatters) | **Considered; superseded by the chosen approach.** Sound and the "right" architectural home, but changing the fusion predicate in `Dialect/Flow/Transforms/RegionOpUtils.cpp` (1015 lines, shared by **all** backends) has very high blast radius. The chosen canonicalization achieves the same single in-place write with a tiny, local, provably-correct rewrite. |
| 4 | Model-side workaround (in-place `fill_()` on strided views) | **Not a compiler fix.** Sound unblock; already used as a workaround in the downstream model. Paired with this compiler fix. |
| 5 | Relax `VerifyWorkgroupDistribution` | **Rejected outright.** The verifier is load-bearing for GPU correctness. |

## Decision: a flow-stage tensor canonicalization

**The chosen fix collapses the two axis-disjoint writes into a single in-place
multi-strided write *before* dispatch lowering, using a local tensor identity.**

### The identity

Writing `src` into `inter` at an *inner* slice, then that result into `base`
at an *outer* slice, is equivalent to writing `src` directly into `base` at the
composed slice, **whenever `inter` is value-equivalent to
`extract_slice(base)` at the outer slice**:

```mlir
%inner = tensor.insert_slice %src into %inter[innerOff, innerSize, innerStride]
%outer = tensor.insert_slice %inner into %base[outerOff, outerSize, outerStride]
   ==>
%new   = tensor.insert_slice %src into %base[outerOff + innerOff*outerStride, innerSize, outerStride*innerStride]
```

For the reproducer this composes `insert[0,0][H/2,W/2][1,2]` (into the
even-rows intermediate) + `store[0,0][H/2,W][2,1]` (to the output) into a
single `insert[0,0][H/2,W/2][2,2]` — exactly the in-place `[2,2]` form.

### Why this is correct

- The composed write touches exactly the base positions `src` would have
  reached through the two-step write
  (`outerOff + (innerOff + k*innerStride)*outerStride`), and leaves every
  other base position unchanged.
- It is valid whenever `inter == extract_slice(base)` at the outer slice. The
  pattern recognizes two provable cases:
  1. **Extract form** — `inter` *is* `tensor.extract_slice %base` with the same
     offsets/sizes/strides as `outer` (checked with `isSameAs`).
  2. **Uniform-fill form** — both `inter` and `base` are `linalg.fill`s of the
     same scalar value. (This is the form the torch-lowered reproducer actually
     reaches: the even-rows intermediate is `linalg.fill false` because the
     constant-zeros base's strided `extract_slice` folds to a fill. By value,
     that fill equals the base's strided subregion.)

### Why this is the right layer

- The divergence is born at flow formation, so collapsing it before dispatch
  lowering is the principled choice (same intent as Idea 3) but expressed as a
  tiny, local, obviously-correct rewrite rather than a predicate change in the
  shared fusion heuristics.
- It needs no changes to the verifier, to tiling, or to bufferization.

## Design of `FoldNestedInsertSlice`

Implemented in
`compiler/src/iree/compiler/Dialect/Flow/Transforms/Canonicalize.cpp` and
registered in `CanonicalizePass::initialize` alongside the existing Flow
canonicalization patterns. Small helpers support it:
`getUniformFillValue`, `areEqualScalarValues`, and `composeInsertSliceOffset`.

- **Match:** `outer = tensor.insert_slice` whose source is an `inner =
  tensor.insert_slice`; rank-equal (bails on rank-reduced cases).
- **Equivalence gate:** `inner.dest` (`inter`) must be value-equivalent to
  `extract_slice(base)` at `outer`'s slice (extract form *or* uniform-fill form,
  above).
- **Compose:** requires **static strides** (bails otherwise); combined offsets
  via `affine::makeComposedFoldedAffineApply` (constant-folds the common
  all-zero-offset case), combined sizes are the inner sizes, combined strides are
  the elementwise product.
- **Replace:** a single `tensor.insert_slice %inner.source into %base` at the
  composed slice. The inner insert and the intermediate become dead and are DCE'd
  by the greedy driver.

### Where it fires

`CanonicalizePass` is wired into the DispatchCreation pipeline and is first
invoked before dispatch regions form (so the chain is still at function scope).
Empirically the fold is observable at the **earliest** at
`--compile-to=dispatch-creation` (absent at `--compile-to=global-optimization`;
identical/redundant at `--compile-to=flow`):

| `--compile-to=` | `strides=[2,2]` present? | `tensor.insert_slice` chain? |
|---|---|---|
| `global-optimization` | no | still present |
| `dispatch-creation` | **yes** | collapsed |
| `flow` | yes (redundant) | collapsed |

## Things tried during implementation

This section records the dead ends and course corrections, in order.

### 1. A standalone linalg harness that *didn't* reproduce the bug

To de-risk before touching the compiler, three hand-written linalg modules were
tried as proxies for the torch reproducer:

- `m[0::2,0::2]=True` as a **single** `tensor.insert_slice … [2,2]` →
  **compiled**. Decisive: it proved an in-place `[2,2]` store is a valid,
  compilable target shape.
- The same write as **two nested** `insert_slice`/`extract_slice` on different
  axes → **also compiled** (flow *split* it into two single-axis dispatches).
- The same write with a `linalg.fill` intermediate (matching the torch-lowered
  form) → **also compiled** (again split).

**None of them reproduced the failure.** The reason: the torch path lowers
`m[0::2,0::2]=True` through `aten.slice_scatter` (not `tensor.extract_slice`),
and at dispatch-formation time the intermediate is a separate `linalg.fill`
(not a view of the base). Flow handles the extract-based forms by splitting,
but fuses the slice_scatter/fill form into the transposed read-modify-write
dispatch. **Lesson:** a faithful reproducer must go through the real torch
frontend; hand-written linalg proxies can mislead. The only reliable test
harness for the fix was the torch `reduced_reproducer.mlir` itself (one build
iteration per change).

### 2. Why it is not an MLIR-internal fold

The identity is a general tensor property, so it is fair to ask why it does not
live in MLIR's `tensor::InsertSliceOp` canonicalization — specifically next to
the existing `foldInsertAfterExtractSlice`
(`third_party/llvm-project/mlir/lib/Dialect/Tensor/IR/TensorOps.cpp`). Three
reasons, in increasing order of importance:

**It is a different *kind* of fold.** `foldInsertAfterExtractSlice` is a
round-trip **identity elimination**: given
`%e = tensor.extract_slice %X[slice]` and `%r = tensor.insert_slice %e into
%X[slice]` with the *same* offsets/sizes/strides and
`extract.source == insert.dest`, it returns `%X` (the insert is a no-op). Our
case is a **composition**: an inner insert at one slice (`[1,2]`) nested inside
an outer insert at a *different* slice (`[2,1]`). The two slices differ, so it
is structurally not a round-trip and that fold cannot match it — loosening its
"same slice" check does not help, because the point is to *combine* two
different slices into one.

**No existing MLIR fold composes this pattern.** Three MLIR mechanisms touch
nested insert/extract; none covers a sub-region insert through a
value-equivalent intermediate:

- `foldInsertAfterExtractSlice` (`TensorOps.cpp:2956`) — the same-slice
  round-trip identity above (works for **any** strides; the `[1,1,1,1]` in its
  doc comment is just an example, not a restriction). A no-op elimination, not a
  composition.
- `foldInsertAfterInsertSlice` (`TensorOps.cpp:2944`) — nested
  insert-of-insert, but it only *redirects the destination* (re-points the outer
  insert at the inner's dest); it does not compose the two slices.
- `InsertSliceOfInsertSliceFolder` (`FoldTensorSubsetOps.cpp:180-222`, via the
  `populateMergeConsecutiveInsertExtractSlicePatterns` the Flow
  `CanonicalizePass` already pulls in) — the closest of the three: it *does*
  attempt to compose two nested inserts (`mergeOffsetsSizesAndStrides`). But it
  requires **matching sizes** and bails when they differ — exactly our case: the
  inner insert writes a `[H/2,W/2]` sub-region while the outer writes the full
  `[H/2,W]` intermediate, so the size check fails (a copy would be needed) and
  nothing folds. (It is not a stride restriction.)

The composition this fix needs is the **mismatched-size** case: a sub-region
insert through an intermediate that is merely *value-equivalent* to a strided
view of the base (an `extract_slice`, or a uniform `linalg.fill` of the same
value). That **value-equivalence gate** is precisely what MLIR's matching-size
design cannot express, and is the real reason a new fold is required.

**Even a correct MLIR composition fold would not fix the real bug (layering).**
To handle the form that actually occurs in the torch-lowered IR — where the
intermediate is a `linalg.fill`, not an `extract_slice` — the fold must inspect
`linalg::FillOp`. The `tensor` dialect cannot depend on `linalg` (linalg
depends on tensor; the reverse is a cycle), so this cannot live in
`tensor::InsertSliceOp` canonicalization. The pure `extract_slice` form it
*could* handle does not survive in practice: the even-rows `extract_slice` of
the constant-zeros base is itself folded to a `linalg.fill` before the
composition would fire, and greedy pattern ordering offers no guarantee the
composition would win that race anyway.

For all three reasons the fold lives in IREE's Flow `Canonicalize.cpp`, which
already depends on both `tensor` and `linalg`. An upstream MLIR follow-up is
still worthwhile, but it would need a generic "uniform-fill" *interface*
(`linalg::FillOp` + `tensor::SplatOp` + constant splats, behind something like
a `UniformTensorValueOpInterface`) so the `tensor` dialect can recognize uniform
intermediates without taking a `linalg` dependency.

### 3. Build failures fixed during the first compile

The initial pattern had three compile errors, each corrected:

- `linalg::FillOp` has no `getValue()` → use `fillOp->getOperand(0)` (operand 0
  is the fill value).
- `tensor::SplatOp::getOperand()` requires an index → the splat branch was
  dropped (the reproducer uses `linalg.fill`, so it was unneeded).
- `MLIRContext::getAffineConstantExpr` does not exist → it is a `Builder`
  method: `builder.getAffineConstantExpr(outerStride)`.

### 4. An "insurance" canonicalize stage that was added then reverted

Because the fold must fire before `ConvertDispatchRegionsToWorkgroupsPass`
(which turns the fills into dispatch-tensor loads, after which the intermediate
is no longer a view of the base), a second change was tried: an extra
`createCanonicalizePass` inserted into `DispatchCreation/Passes.cpp` between
dispatch region formation and `ConvertDispatchRegionsToWorkgroups`.

It was **reverted.** It broke two `DispatchCreation/test/pipeline_tests*`
FileCheck tests (the new canonicalize stage more-aggressively canonicalized
equivalent IR, changing dispatch-argument shape), and it turned out to be
unnecessary: the fold already fires at the canonicalize that runs **before**
dispatch region formation, where the two-insert chain is still at function
scope. Verified by removing the insurance and re-running: the reproducer still
passes and the two pipeline tests pass again. The shipped change is therefore a
**single file** (`Canonicalize.cpp`, +118 lines).

## Validation

All run against the rebuilt tree (`/Users/alex/Developer/.iree-build`, Ninja,
RelWithDebInfo, assertions on):

- `reduced_reproducer.mlir` (was exit 1, both oracle errors) → **exit 0**; all
  single-stride controls and the doubly-strided `variant_only_line16.mlir`
  still pass.
- Flow-stage / `--compile-to=dispatch-creation` dispatch body is a single
  `dispatch.tensor.store ... strides=[2, 2]` (the fold fired; no transposed
  read-modify-write).
- **Full compiler ctest suite (`iree/compiler/`, 1153 tests): 1153/1153 pass,
  0 failures**, including `verify_workgroup_distribution`,
  `tile_and_distribute_workgroups_using_forall`, `canonicalize`, and the entire
  `Codegen/LLVMCPU` and `Codegen/Common` suites.
- Downstream model: with the previously-failing out-of-place path live, the
  module compiles **and** is numerically correct on `llvm-cpu` (shape, dtype,
  per-pixel disjoint masks, RGGB layout).

### Convergence to the workaround's dispatch shape

A two-path comparison (in-place `fill_()` workaround vs out-of-place `=`) at
`--compile-to=dispatch-creation` shows that, with the fix, **both paths lower to
textually identical dispatches**: four single-dispatch masks, each a single
in-place `dispatch.tensor.store ... strides=[2, 2]` at the Bayer cell offsets
`[0,0]/[0,1]/[1,0]/[1,1]`. The only residual difference is the fill op
(`linalg.generic { yield %true }` vs `linalg.fill ins(%true)`), which is
cosmetic. The fix therefore recovers the same optimal dispatch shape the
workaround produced — not merely a "compiles" outcome.

(This corrects an earlier hypothesis that the in-place path would lower to
single-axis `[1,2]`/`[2,1]` stores while the out-of-place path lowered to
`[2,2]`. In the real torch-lowered pipeline, both produce `[2,2]`.)

## Risks, limitations, and follow-ups

- **Static strides only.** The fold composes strides only when both are static.
  Dynamic strides fall through (no change in behavior). This covers the
  practical cases (strided writes have compile-time strides) without risking
  incorrect affine composition.
- **Blast radius is intentionally small.** The pattern matches only the narrow
  nested-insert-through-equivalent-intermediate shape; the full compiler test
  suite confirms no regressions. Because it is registered in `CanonicalizePass`,
  it runs wherever `iree-flow-canonicalize` runs — appropriate, since this is a
  general tensor identity.
- **Upstream MLIR fold.** Generalizing this into MLIR's `tensor` canonicalization
  (behind a uniform-fill interface to avoid the linalg layering issue) would
  benefit all MLIR users and is the natural upstream home.
- **Secondary bugs deferred.** The `2^52` scratch alloca and the
  `cpuAllocationFn` descriptor-memory-space preservation are independent issues
  observed in the (now-unreached) failing path; they should be filed and fixed
  separately. With this canonicalization they no longer surface for this
  workload.
- **Single-dispatch vs split.** This fix produces a single in-place `[2,2]`
  dispatch per mask. The long-term "Idea 1" codegen work (distributing a
  transposed store) is only relevant if a future pattern produces a transposed
  dispatch that cannot be canonicalized away; this fix removes that need for the
  common doubly-strided writeback.

## References

- Fix: `compiler/src/iree/compiler/Dialect/Flow/Transforms/Canonicalize.cpp`
  (`FoldNestedInsertSlice` + helpers; registered in `CanonicalizePass::initialize`).
- Verifier (unchanged, load-bearing):
  `compiler/src/iree/compiler/Codegen/Common/VerifyWorkgroupDistribution.cpp`.
- Pipeline wiring (unchanged):
  `compiler/src/iree/compiler/DispatchCreation/Passes.cpp`
  (`CanonicalizePass` is part of the DispatchCreation pipeline).
