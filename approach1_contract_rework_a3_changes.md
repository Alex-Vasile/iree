# Reasoning Companion — A3 Polish Changes (per-finding changelog)

> **Purpose:** durable per-finding record of every change A3 made to
> `approach1_contract_phase1_plan.md` (v2 → v3). For each finding: the **OLD**
> text (A2 v2), the **NEW** text (A3 v3, as it now reads in the plan), and the
> `file:line` evidence that justifies it. The *decision rationale* lives in the
> **Reasoning & Justification** section at the end; the per-finding blocks are the
> diff/evidence trail.
>
> **Scope:** R2's verdict was **CONVERGED**. This pass fixes R2's four remaining
> *execution-risk* items (m1–m4). It does **NOT** re-open the settled mechanism
> (insert_slice as a `TilingInterface` anchor tiled via INITIAL tiling, stride
> from its own `strides` attribute), the verified call chain, the `resultSizes =
> iterSizes` formula, or the scatter-vs-offset gate design — all final.
>
> **Grounding rule:** every `file:line` below was opened personally in the
> `~/Developer/iree` / `third_party/llvm-project` tree this session. R2's cites
> were re-verified (one — the m2 pass name — was found stale and corrected).
> `[INFERENCE]` marks reasoning-only claims. No code changed, no builds run.

---

## Summary

| # | Finding (R2 ref) | Kind | Severity | R2's own verdict |
|---|---|---|---|---|
| m1 | §4a marker-gate names an impossible gate point (`getIterationDomain` cannot fail) | **FIX (must)** | MEDIUM | execution risk, self-correcting |
| m2 | §8 EXEC pipeline missing the memref→LLVM pass; R2's `-convert-memref-to-llvm` name is itself stale | **FIX (must)** | LOW-MED | execution risk, self-correcting |
| m3 | G4a 1-iteration forall is foldable → G3 FileCheck fragility | **FIX** | LOW | execution risk, IR-only |
| m4 | §4a containment over-stated (best-effort, not hard) | **REWORDING (honesty)** | LOW | execution risk |

**Net of the pass:** every must-fix is resolved with codebase evidence; the plan
is execution-ready (no remaining must-fixes). Two of the four (m2, m4) are
sharper than R2 framed: **m2** corrects R2's *own* stale cite (the pass R2 named
does not exist in this checkout), and **m4** grounds a new limitation R2 only
flagged as a checkpoint (the partial-forall leak).

---

## m1 — §4a marker-gate names an impossible gate point (MUST-FIX)

**OLD (v2, §4a firewall 1):**
> The new `InsertSliceOpTiling` model's `getIterationDomain` /
> `getTiledImplementation` / `getResultTilePosition` early-return `failure()`
> unless a discardable marker … is present. … the bare
> `dyn_cast<TilingInterface>` … now *succeeds*, but the very next call
> (`getIterationDomain`) returns `failure()` → the consumer skips it.

**NEW (v3, §4a firewall 1):**
> The marker-gate lives on the two impl methods that can express failure —
> **`getTiledImplementation`** (returns `FailureOr<TilingResult>`,
> `TilingInterface.td:107`) and **`getResultTilePosition`** (returns
> `LogicalResult`, `:149`). It does **NOT** live on `getIterationDomain`: that
> method returns `SmallVector<Range>` (`:80-85`, default `return {}`) — there is
> no `LogicalResult` to return, so it reports the **true source-shape domain
> unconditionally**. … **[full verified bail path re-traced]** … `tileToThreads`
> bails silently (`GPUGreedilyDistributeToThreads.cpp:96-98`). **No strided
> writeback is ever emitted** — the `parallel_insert_slice` at `:619-621` is
> downstream of the failing call and is never reached.

**Evidence (all opened this session):**
- `include/mlir/Interfaces/TilingInterface.td:80-85` — `getIterationDomain`
  `/*retTy=*/"::mlir::SmallVector<::mlir::Range>"`, `defaultImplementation=
  "return {};"`. **No `LogicalResult`.** A literal "gate it on the marker" is
  ill-formed. (PadOp's model at `TensorTilingInterfaceImpl.cpp:33-44` confirms the
  non-failing shape.) ✓
- `TilingInterface.td:107` — `getTiledImplementation`
  `/*retType=*/"::mlir::FailureOr<::mlir::TilingResult>"`, `defaultImplementation
  [{ return {}; }]` (empty `FailureOr` = failure). **Can express failure.** ✓
- `TilingInterface.td:149` — `getResultTilePosition`
  `/*retType=*/"::llvm::LogicalResult"`, `defaultImplementation [{ return
  failure(); }]`. **Can express failure.** ✓

**Verified bail path (the containment proof):**
1. Entry: `GPUGreedilyDistributeToThreads.cpp:139`
   `if (auto tilableOp = dyn_cast<TilingInterface>(op))` inside the `processRegion`
   walk (`:114-154`); routed to `tileToThreads` at `:145`. ✓
2. `tileToThreads` (`:43-44`) → `scf::tileConsumerAndFuseProducersUsingSCF`
   (`:93-95`) → `tileUsingSCF`.
3. `tileUsingSCF` calls `getIterationDomain` (valid ranges, **ungated** — per m1
   this is correct), sets up the body lambda (`TileUsingInterface.cpp:1158-1226`),
   and calls `generateLoopNest` (`:1241`).
4. `generateLoopNestUsingForallOp` creates the forall
   (`scf::ForallOp::create` at `:585`), then invokes the body lambda via
   `tiledBodyFn` (`:607`).
5. The body lambda calls the static helper `getTiledImplementation`
   (`TileUsingInterface.cpp:820-829`) → on `FullReduction` (the strategy for a
   non-reduction op like `insert_slice`) forwards to the interface method
   `op.getTiledImplementation(rewriter, offsets, sizes)` (`:829`). On an
   **unmarked** op the gated impl returns failure → `:1197-1199`
   `rewriter.eraseOp(clonedOp); return op.emitOpError("failed to tile
   operation");` (a `failure()` LogicalResult). ✓
   *(Symmetric second gate: `getResultTilePosition` failure → `:1215-1220`
   `notifyMatchFailure("failed to get slice of result produced")`.)*
6. The lambda failure propagates: `generateLoopNestUsingForallOp:607-610`
   `if (failed(tiledBodyFn(...))) return rewriter.notifyMatchFailure(loc,
   "failed to generate loop body");` → `tileUsingSCF:1244-1245`
   `if (failed(loopsOr)) return op.emitOpError("failed to generate tiling
   loops");`. ✓
7. `tileConsumerAndFuseProducersUsingSCF` fails → `tileToThreads` silent return
   (`GPUGreedilyDistributeToThreads.cpp:96-98`). ✓

**Effect:** the cast succeeds but tiling execution fails fast; the
`parallel_insert_slice` writeback (`:619-621`) is *downstream* of the failing
`getTiledImplementation`/`getResultTilePosition` call and is **never reached**.
The containment §4a wants holds; only the *named gate site* was wrong.

---

## m2 — §8 EXEC pipeline missing the memref→LLVM pass; pass name corrected (MUST-FIX)

> **A3 found R2's own cite stale and corrected it.** R2 §3.2 named the missing
> pass `-convert-memref-to-llvm`. **No such pass exists in this checkout.** The
> real pass is `-finalize-memref-to-llvm`. Every `mlir/test` pipeline uses the
> latter; a grep for `-convert-memref-to-llvm` across `mlir/test` returns zero
> matches.

**OLD (v2, §8 explicit pipeline block):**
```bash
mlir-opt %s -transform-interpreter \
  -one-shot-bufferize="allow-return-allocs bufferize-function-boundaries" \
  -convert-linalg-to-loops \
  -convert-scf-to-cf -convert-cf-to-llvm \
  -convert-arith-to-llvm -convert-math-to-llvm \
  -convert-func-to-llvm -convert-index-to-llvm \
  -reconcile-unrealized-casts -o %t
```
(and the §8 prose/§11 risk #2 cited the non-existent `-convert-memref-to-llvm`
three times, framing the strided copy as "could be dropped or rejected
`[INFERENCE]`".)

**NEW (v3, §8 pipeline block):**
```bash
mlir-opt %s -transform-interpreter \
  -one-shot-bufferize="allow-return-allocs bufferize-function-boundaries" \
  -convert-linalg-to-loops \
  -convert-scf-to-cf \
  -expand-strided-metadata -finalize-memref-to-llvm \
  -convert-cf-to-llvm \
  -convert-arith-to-llvm -convert-math-to-llvm \
  -convert-func-to-llvm -convert-index-to-llvm \
  -reconcile-unrealized-casts -o %t
```
plus corrected pass names in §8 step 3, the §8 "G4 risk" paragraph, and §11 risk
#2, and R2 §4's resolved-positive finding folded in (the strided copy is **not**
rejected).

**Evidence (all opened this session):**
- `mlir/include/mlir/Conversion/Passes.td:994-995` — `def
  FinalizeMemRefToLLVMConversionPass :
  Pass<"finalize-memref-to-llvm", "ModuleOp">`. The pass IS named
  `finalize-memref-to-llvm`. ✓
- `lib/Conversion/MemRefToLLVM/MemRefToLLVM.cpp:1140` —
  `class MemRefCopyOpLowering : public ConvertOpToLLVMPattern<memref::CopyOp>`;
  registered into the pattern set at `:2109`
  (`patterns.add<...MemRefCopyOpLowering>(...)`) — i.e. **under
  `-finalize-memref-to-llvm`**, the pass R2 omitted. ✓
- `MemRefToLLVM.cpp:1257-1278` — `MemRefCopyOpLowering::matchAndRewrite`: the
  contiguity fork at `:1263-1272` (`isContiguousMemrefType`); `:1274-1275`
  contiguous×contiguous → `lowerToMemCopyIntrinsic` (flat `LLVM::MemcpyOp`);
  **`:1277` anything strided → `lowerToMemCopyFunctionCall`** (generic
  element-wise runtime copy honoring both operands' layouts). **The strided copy
  is NOT rejected** — this is R2 §4, now personally re-verified. ✓
- `test/Integration/Dialect/Complex/CPU/correctness.mlir:1-11` — the full CPU
  runner pipeline template the §8 block is modeled on:
  `-one-shot-bufferize="bufferize-function-boundaries" --canonicalize \
  -convert-scf-to-cf ... -finalize-memref-to-llvm -convert-math-to-llvm ... \
  -convert-vector-to-llvm ... -convert-func-to-llvm -convert-arith-to-llvm \
  -convert-cf-to-llvm -reconcile-unrealized-casts | mlir-runner`. Shows
  `-finalize-memref-to-llvm` positioned **after `-convert-scf-to-cf`**, before
  the `-convert-*-to-llvm` dialect conversions. ✓
- `test/Conversion/MemRefToLLVM/expand-then-convert-to-llvm.mlir:1` — the
  canonical **strided-memref** pair: `-expand-strided-metadata
  -finalize-memref-to-llvm …`. Justifies adding `-expand-strided-metadata`
  alongside the finalize pass (the G4 writeback produces a strided
  `memref.subview`). ✓
- `test/Integration/Dialect/Vector/CPU/transfer-write.mlir:1-4` — the
  monolithic `-test-lower-to-llvm | mlir-runner` **fallback**, unchanged. ✓

**Position rationale:** `-finalize-memref-to-llvm` lowers `memref.subview` /
`memref.copy` (the bufferization output of the `parallel_insert_slice`
writeback, §8 step 2). It must run after `-convert-scf-to-cf` / `-convert-linalg-to-loops`
(which *produce* those memref ops) and before `-reconcile-unrealized-casts`. The
`correctness.mlir` template fixes the ordering; without the pass,
`memref.copy`/`memref.subview` survive into LLVM-dialect conversion and are
rejected as illegal ops — a **COMPILE-fail unrelated to strides** (R2 §3.2).
`-expand-strided-metadata` precedes it per the strided-memref template.

---

## m3 — G4a 1-iteration forall is foldable → G3 FileCheck fragility

**OLD (v2, §3 G3 IR FileCheck):**
> assert `parallel_insert_slice %tile into %o0[0,0][2,4][2,1]` … (Minor risk: a
> 1-iteration `scf.forall` *could* be folded by canonicalization before the
> check; if so, use the 2-tile genuine scatter …).

**NEW (v3, §3 G3 IR FileCheck):**
> the load-bearing tokens are the `[2,4]` sizes (== source) AND `[2,1]` strides
> *together*. … if [the 1-iteration forall] folds, the writeback surfaces as a
> plain `tensor.insert_slice` (terminator rewrite) rather than
> `tensor.parallel_insert_slice`, and the dest may switch from the region iter
> arg `%o0` to the folded target — so a FileCheck anchored on the op name or on
> `%o0` would be a **spurious RED under folding, not a mechanism failure**. Match
> the **geometry under EITHER op mnemonic**, e.g.
> `// CHECK: tensor.{{(parallel_)?}}insert_slice {{.*}}[0, 0] [2, 4] [2, 1]`. …
> **Fallbacks** if even the geometry regex misses: (i) drop `-canonicalize` from
> the IR-check `mlir-opt` run, or (ii) use the 2-tile genuine scatter
> (`source<4x4> → dest<8x4>`, `tile_sizes [2,4]` → 2 non-foldable tiles — the G4b
> shape, §7). G4 (EXEC) is unaffected.

**Evidence / reasoning:**
- Sub-case A tiles a `<2x4>` source with `tile_sizes [2,4]` → exactly **1
  iteration**; a single-iteration `scf.forall` is a canonicalization target, and
  folding it rewrites the terminator `parallel_insert_slice` → a standalone
  `insert_slice` (R2 §3.3). Anchoring the FileCheck on the op mnemonic or on the
  region iter arg `%o0` then misses. R2 §3.3 / risk #3 prescribes matching the
  geometry under either form.
- **Why the geometry line is stable under the fold** `[INFERENCE]`: the writeback
  is a genuine `[2,1]` scatter (tile row 1 → dest row 2), which is **not**
  copy-elidable — the verifier (`TensorOps.cpp:2885-2896`) checks sizes/in-bounds,
  and a strided scatter into distinct dest rows has no identity-fold target. So
  the `[0,0][2,4][2,1]` geometry survives canonicalization; only the op mnemonic
  / dest operand change. The regex `{{(parallel_)?}}` absorbs that change in one
  self-contained assertion — the cleanest option (vs. two `CHECK` prefixes, or
  disabling canonicalization globally).
- **Why G4 (EXEC) is unaffected:** EXEC consumes the *pre*-canonicalization IR
  through the full lowering pipeline (§8); the fold, if it occurs, is a property
  of the IR-check `mlir-opt` invocation only.

---

## m4 — §4a containment honesty (best-effort, not hard)

**OLD (v2, §4a):** stated the two firewalls as containing the global interface
change, with the marker-gate framed as the PRIMARY firewall and the allow-list as
"defense-in-depth." Did not state what slips through.

**NEW (v3, §4a — new "Containment limits — best-effort, not hard" note):**
> firewall 1 is a *behavioral* gate, not a type-system guarantee; it does not, on
> its own, hard-contain the global interface change. What it does NOT catch:
> - **A partial/empty `scf.forall` body may be left in the IR on an unmarked
>   hit.** The gate fires *inside* the body lambda (`TileUsingInterface.cpp:607`),
>   which runs *after* the forall shell is already created
>   (`scf::ForallOp::create :585`); `notifyMatchFailure` (`:610`) logs the bail
>   but does **not** erase that forall. Whether the caller rolls it back is an
>   **execution checkpoint, not a guarantee** (R2 risk #4).
> - **A consumer that bypasses the gated methods.** Any pass that builds a tiled
>   body for `insert_slice` *without* routing through
>   `getTiledImplementation`/`getResultTilePosition` … is not contained by the
>   marker.
> - **`tileToThreads`'s silent bail is a property of *current* code**
>   (`GPUGreedilyDistributeToThreads.cpp:96-98`). If a future change made it
>   hard-error or stopped routing through `getTiledImplementation`, an unmarked
>   `insert_slice` would be mis-tiled.
> The **hard** firewall is therefore firewall 2 (the IREE allow-list), which MUST
> ship in the integration PR before any marker is dropped.

**Evidence (opened this session):**
- `TileUsingInterface.cpp:585-586` — `forallOp = scf::ForallOp::create(rewriter,
  loc, ...)` (the forall shell is materialized in the IR). ✓
- `TileUsingInterface.cpp:607-610` — the gate fires *inside* `tiledBodyFn`, which
  runs **after** `:585`; `return rewriter.notifyMatchFailure(loc, "failed to
  generate loop body");` logs the failure but performs **no erase** of `forallOp`
  (which was already pushed to `loops` at `:594`). The rollback is therefore a
  caller responsibility — R2 risk #4's checkpoint. ✓
- `GPUGreedilyDistributeToThreads.cpp:96-98` — `if (failed(tiledResults)) {
  return; }` ("returns silently"); best-effort *today*, not a contract. ✓
- R2 §3.4 — the allow-list (firewall 2) is the real integration firewall; the
  marker gate is staged-rollout, not load-bearing containment on its own.

**Honest scoping:** the note names three concrete escape hatches and designates
the allow-list as the hard boundary. It does **not** claim the marker makes the
global change safe to ship unattended.

---

## Reasoning & Justification (the WHY of each A3 change)

> Per-finding OLD/NEW/evidence: the blocks above. This section is the *decision
> rationale* — why each fix takes the shape it does, and why the settled design
> was left untouched.

**m1 — why the gate moved off `getIterationDomain`, and why that still contains.**
`getIterationDomain`'s signature is fixed by the interface contract
(`SmallVector<Range>`, `TilingInterface.td:80-85`) — it has no failure channel,
so "gate it on the marker" is not just suboptimal, it is unimplementable. The two
methods that *can* fail are `getTiledImplementation` (`:107`, `FailureOr`) and
`getResultTilePosition` (`:149`, `LogicalResult`). Moving the gate there is not a
weakening: the bail path I re-traced shows the gate fires *before* the
`parallel_insert_slice` writeback is ever constructed (`:619-621` is downstream of
the failing `:1193-1199`/`:1211-1220` call), so an unmarked `insert_slice` still
emits **zero** strided IR. The containment §4a wants is preserved; only the named
site was wrong. `getIterationDomain` is left ungated and returns the true
source-shape domain — exactly as R2 §5.1 prescribed. This is a precision fix to
descriptive text; the mechanism is untouched.

**m2 — why the pass name matters, and why R2's `[INFERENCE]` could be discharged.**
R2 was directionally right (a memref→LLVM pass is missing) but cited a pass that
does not exist in this checkout. `-convert-memref-to-llvm` was the historical
name; the current pass is `-finalize-memref-to-llvm` (`Passes.td:994-995`), and
`MemRefCopyOpLowering` (`MemRefToLLVM.cpp:1140`, reg `:2109`) lives under it.
Because the very pass being added is the one R2 §4 analyzed, I could re-verify
R2's finding personally (`:1263-1277`: strided → `lowerToMemCopyFunctionCall`, an
element-wise runtime copy, **not** a rejection) and fold it in — turning the
§8/§11 "could be dropped or rejected `[INFERENCE]`" into a grounded
"lowers correctly (slowly)." This is still honest: end-to-end *cell correctness*
remains G4-proven only; what is discharged is the *compile-path* half of the
fear. Adding `-expand-strided-metadata` alongside (per the
`expand-then-convert-to-llvm.mlir` template) matches the canonical strided-memref
pipeline, which is exactly the G4 shape. The `-test-lower-to-llvm` fallback stays
(transfer-write.mlir).

**m3 — why a single regex beat two prefixes or disabling canonicalization.**
R2 offered three options (match geometry under either op; disable CSE; fall back
to the 2-tile shape). A single FileCheck line
`tensor.{{(parallel_)?}}insert_slice {{.*}}[0, 0] [2, 4] [2, 1]` absorbs the only
real canonicalization effect (the terminator rewrite
`parallel_insert_slice`→`insert_slice` on a folded 1-iteration forall) while
keeping the load-bearing `[2,4]`+`[2,1]` tokens asserted. It is self-contained,
needs no extra run line, and is the least-likely to mask a real regression. The
no-canonicalize and 2-tile options stay as documented fallbacks for the residual
"geometry itself rewritten" risk `[INFERENCE]` (a strided scatter should not
elide, but that is verifier-grounded, not run-confirmed). G4 is unaffected
because EXEC never sees the folded IR.

**m4 — why containment was stated as best-effort, and where the hard line is.**
A marker-gated impl is a *behavioral* brake, not a type-system guarantee: it only
contains consumers that route through `getTiledImplementation`/
`getResultTilePosition`. The m1 bail-path trace surfaced a concrete gap R2 had
only flagged as a checkpoint — the gate fires *inside* the body lambda
(`TileUsingInterface.cpp:607`), **after** the forall shell is created (`:585`),
and `notifyMatchFailure` (`:610`) does not erase it, so an unmarked hit can leave
a stray empty `scf.forall` unless the caller cleans up. Stating this plainly (and
naming the bypass-via-ungated-methods and best-effort-`tileToThreads`-bail
escape hatches) keeps the integration PR honest: the marker is staged-rollout
scaffolding; the **hard** firewall is the IREE allow-list (firewall 2), which must
land before any marker is dropped. Over-claiming containment here would be the
most dangerous kind of polish error — a false sense of safety on the GO-authorized
integration path.

---

## Note on the settled mechanism — NOT re-litigated

Per the assignment, the chosen mechanism (insert_slice as a `TilingInterface`
anchor tiled via INITIAL tiling, stride source = the insert_slice's own `strides`
attribute) and the verified call chain
(`tile_using_forall` → `tileToForallOpImpl` → `dyn_cast<TilingInterface>`
→ `tileUsingSCF` → body lambda → `getResultTilePosition` (new `resultStrides`)
→ writeback `:616-617`) are final. A3 touched only the **descriptive precision**
of four execution-risk sites (the §4a gate point, the §8 pipeline + pass name, the
§3 FileCheck robustness, the §4a containment claim) — not the mechanism, the size
math (`resultSizes = iterSizes`), or the scatter-vs-offset gate design.
