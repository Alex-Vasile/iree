# Follow-ups — doubly-strided writeback canonicalization

Tracking file for the `FoldNestedInsertSlice` change in
`compiler/src/iree/compiler/Dialect/Flow/Transforms/Canonicalize.cpp`
(see design doc: `doubly-strided-writeback-llvm-cpu.md`).

This is an engineering TODO list, **not** part of the published docs site (it is
deliberately not wired into the mkdocs nav).

---

## P0 — Tests for the new pass (currently the biggest gap)

The pass shipped with **no dedicated lit test**. The full compiler ctest suite
passing (1153/1153) proves *no regression*, but nothing positively asserts the
fold fires or that its edge cases behave. Add tests to
`compiler/src/iree/compiler/Dialect/Flow/Transforms/test/canonicalize.mlir`
(preferred — it already exercises `iree-flow-canonicalize`) using
`--split-input-file`, with these cases:

- [ ] **Doubly-strided, uniform-fill intermediate (the main case):**
      `%base = linalg.fill %v0`, `%inter = linalg.fill %v0` (same value),
      `%inner = insert_slice %src into %inter[0,0][sz][1,2]`,
      `%outer = insert_slice %inner into %base[0,0][sz2][2,1]`
      → expect a single `insert_slice %src into %base[0,0][sz][2,2]`.
- [ ] **Extract form:** `%inter = tensor.extract_slice %base[0,0][sz2][2,1]`
      (same slice as the outer insert) → composes to `[2,2]`.
- [ ] **Different fill values (negative):** `%inter = fill %v0`, `%base = fill %v1`
      (v0 ≠ v1) → **no fold** (must keep both inserts).
- [ ] **Non-uniform intermediate (negative):** `inter` is a real compute op, not
      a fill/extract → **no fold**.
- [ ] **Single insert (negative):** no nested insert → unchanged.
- [ ] **Rank-reduced (negative):** `inter` rank < `base` rank → **no fold** (bail).
- [ ] **Dynamic offsets:** offsets are `%a, %b` → composes via `affine.apply`
      `(d0,d1)[s0] -> (d0 + d1*s0)`, constant-folds when static.
- [ ] **Dynamic stride (negative):** a stride is `?` → **no fold** (static-stride
      requirement), behavior unchanged.

Acceptance: each case `CHECK`s the expected IR (folded or unchanged); the
negative cases assert the *absence* of a `[2,2]` collapse.

## P1 — Wire the design doc into the docs site

- [ ] Add an entry under the `design-docs` section of
      `docs/website/mkdocs.yml` (nav) for
      `developers/design-docs/doubly-strided-writeback-llvm-cpu.md`, matching the
      style of the existing entries (`cuda-hal-driver`, `function-abi`, etc.).
- [ ] Sanity build the site locally (`mkdocs serve` under `docs/website/`) and
      confirm the page renders and the nav link works.

## P1 — Upstream the fold to MLIR

The identity is general; the only reason it is IREE-local is the `linalg::FillOp`
dependency (see design doc §"Why it is not an MLIR-internal fold").

- [ ] Propose an upstream MLIR composition fold on `tensor::InsertSliceOp` that
      recognizes a uniform intermediate via a generic interface (e.g. a new
      `UniformTensorValueOpInterface` covering `linalg::FillOp`,
      `tensor::SplatOp`, and splat constants) — so the `tensor` dialect can match
      it without a `linalg` dependency.
- [ ] Handle the **extract form** there too (no interface needed), and ensure it
      wins the ordering race against `extract_slice(linalg.fill) → linalg.fill`.
- [ ] Once upstream lands and is pulled in, retire the IREE-local pattern.

## P2 — File/fix the two secondary bugs (observed in the now-unreached path)

Independent of this fix; surfaced while debugging. With the canonicalization
these no longer occur for this workload, but they are real latent defects:

- [ ] **`2^52 × 2^52` scratch alloca:** a dynamic dimension's
      `util.assume.int<umax=…>` bound leaks in as the *static* row stride/size of
      a scratch `memref` (`Codegen/.../IREEComprehensiveBufferize.cpp` /
      allocation type derivation). Trace the leak path and fix.
- [ ] **`cpuAllocationFn` descriptor-space inconsistency**
      (`Codegen/Common/CPU/Passes.cpp:21-38`): unlike `defaultAllocationFn`
      (`IREEComprehensiveBufferize.cpp:56-72`), it does **not** erase
      `#hal.descriptor_type` from scratch allocs — contradicting the documented
      contract. Low-risk consistency fix; shrinks the verifier's blast radius
      (does not by itself fix the primary issue, which this canonicalization
      already resolves).

## P3 — Harden / generalize the pass

- [ ] Add a `verifyInBoundsSlice` bounds check on the composed slice before
      replacing (currently relies on the algebraic proof + RelWithDebInfo
      assertions; a static check is belt-and-suspenders and matches the existing
      `InsertSliceOpConstantArgumentFolder` style).
- [ ] Generalize to **dynamic strides** if a real workload needs it (compose via
      `affine.apply` with the stride as a symbol instead of requiring it static).
- [ ] Consider **rank-reduced** intermediates if they show up in practice.
- [ ] Perf: confirm the single `[2,2]` in-place dispatch has no regression vs the
      single-stride case on representative workloads.

## Cross-repo (editor) — once the fix is upstream/merged

- [ ] `RCDBayerMaskModule` (`src/python/editor/compute/rcd_demosaic.py`) carries
      a regression-harness `if True:`/`if False:` switch between the in-place
      workaround and the formerly-broken out-of-place path. Both now compile to
      textually identical dispatches; once the compiler fix lands upstream, drop
      the workaround branch and the switch.
