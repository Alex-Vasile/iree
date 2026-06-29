# Approach 1 — Iteration Reasoning & Decisions Log

> **Purpose (per user instruction 2026-06-29):** persist ALL justifications and
> reasonings — the main agent's AND every subagent's — to files, so the chain of
> thought is durable, not just the conclusions. This file is the **main agent's**
> reasoning across the arc. Subagent reasoning is in their own files (§6 index).

---

## 1. Session arc (what happened, and why each step)

1. **Inherited a v3.1 plan** (`approach1_phase1_impl_plan.md`) built on the thesis
   that an SCF-only consumer-fusion path could emit a strided writeback without
   the `TilingInterface.td` contract change. That thesis came from an earlier
   advisory ("trust your own stride-source finding").
2. **User asked for an expert evaluation.** I dispatched a senior-compiler
   `oracle` (CompilerExpertReview) with an *adversarial* brief (find holes, verify
   every citation). Reasoning: the plan was about to drive weeks of upstream
   work; a cheap second opinion before execution was high-leverage, and I wanted
   claims re-grounded in source rather than trusted.
3. **The review returned NO-GO with 3 fatal flaws.** Rather than relay it
   unverified, I re-read the three linchpins myself (§2). All three held.
4. **Accepted the NO-GO** and did the cheap, source-of-truth fixes: corrected the
   feasibility doc's §2.1 error, marked the plan SUPERSEDED, corrected memory.
5. **User chose Option 1** (replan around the contract change) and asked for a
   fresh-expert rework/review loop. I designed A1→R1→A2→R2 (§4).

## 2. The refutation — and my own independent verification (the load-bearing part)

The expert review (`expert_review_phase1_plan.md`) gave three fatal flaws. I did
NOT relay them unverified; I re-opened each linchpin:

- **Flaw 1 — `tensor.insert_slice` is not `TilingInterface`.** I read
  `lib/Dialect/Tensor/IR/TensorTilingInterfaceImpl.cpp:311-316` myself:
  `registerTilingInterfaceExternalModels` attaches the interface **only to
  `tensor::PadOp`** (`:314`). CONFIRMED. This refutes the feasibility doc §2.1's
  claim that `insert_slice` "is a `TilingInterface` op and therefore a
  `computeOp`" — a factual error that propagated into the SCF-only plan.
- **Flaw 2 — `fuse_into_containing_op` is producer fusion.** I read
  `lib/Dialect/Linalg/TransformOps/LinalgTransformOps.cpp:1335-1336`: it calls
  `tileAndFuseFirstExtractUse(rewriter, diag, producerOp, containingOp)` —
  **producer** fusion, never `tileAndFuseConsumer`. CONFIRMED. So the plan's
  specified harness wouldn't even exercise the targeted path.
- **Flaw 3 (architectural) — `:2311` reads the candidate, not the consumer.** I
  read `TileUsingInterface.cpp:2479-2519`: `getProducingInsertSliceLikeOp` for a
  `scf.forall` returns `getProducingParallelInsertSlice` (`:2487`) — the loop-
  **internal** `parallel_insert_slice`, which the tiler emits unit-stride
  (`:616`/`:1006`). CONFIRMED. So threading "candidate strides" threads `[1,1]`;
  the external `[2,1]` never enters the consumer-fusion path. **Consumer fusion
  preserves candidate geometry; it cannot create strides.**

**My conclusion (matches the expert + the blocker advisory):** strides can only
be *introduced* at **initial tiling** (`:616-617`/`:447-448`) when the anchor
op's result-to-dest mapping is strided — which requires `getResultTilePosition`
to return strides = **the contract change is a prerequisite, not a fallback.**
The feasibility doc's original HARD+ verdict was right; the v2/v3 "SCF-only"
reframe was the error.

## 3. Decisions taken (each with justification + evidence)

- **Accept NO-GO; do not iterate to v4; do not work around Flaw 1** (e.g. by
  bolting TilingInterface onto insert_slice to rescue consumer fusion).
  *Justification:* Flaw 3 kills it regardless — even with a tileable consumer,
  `:2311` has no stride to propagate. Working around Flaw 1 alone is wasted
  effort. (Blocker advisory concurred.)
- **Correct the doc §2.1 error** (`approach1_tensor_level_fusion.md:84-100`).
  *Justification:* it is a verified-false claim in a durable investigation doc
  that already propagated once. Fixing at the source stops the rot. Evidence:
  `TensorTilingInterfaceImpl.cpp:314` (PadOp-only).
- **Mark the plan SUPERSEDED** (`approach1_phase1_impl_plan.md` head banner).
  *Justification:* leaving a refuted plan unmarked would mislead any reader
  (including future-me). The banner states the 3 flaws + what survives (R2, §3
  offset math) + points to the review.
- **Correct memory.** *Justification:* I had stored a "SCF-only can unblock
  without the contract" fact (`f53dff17a91e3808`) earlier this turn — now refuted.
  I `invalidate`d it and stored the corrected understanding so it cannot poison a
  future session. Also stored a reusable lesson (§5).

## 4. The contract-change rework loop — design rationale

- **Fresh agent per rework AND per review** (per user instruction). *Why:* a
  single agent reviewing its own work carries the same blind spots; fresh agents
  re-derive from source. Each spawn is a distinct `oracle` with the same senior-
  compiler persona.
- **Sequential, not parallel** (A1→R1→A2→R2). *Why:* each step consumes the
  prior artifact on disk (R1 reviews A1's plan; A2 addresses R1's findings).
  No independence to exploit.
- **The crux handed to A1 explicitly** (do not dodge it): linalg indexing maps
  are projected permutations and can't express strides, so a naive
  `resultStrides` out-param is insufficient — the plan must specify the **stride
  source** at initial-tiling time. *Why I foregrounded this:* the prior plan's
  failure was a misidentified stride source; I wanted the new plan forced to
  answer "which value carries the stride, and how does it reach `:616`?" before
  declaring a mechanism.
- **A1 chose mechanism (c): tile `tensor.insert_slice` itself as the anchor**
  via initial tiling; its `getResultTilePosition` returns a strided result
  position sourced from its own strides attribute, flowing to `:616-617`. *My
  read:* this is the right *shape* — it puts the stride source at initial-tiling
  time and uses the slice op's own attribute (the one place `[2,1]` genuinely
  lives). Whether it's *viable* (no-overlap contract, lowering survival, blast
  radius of giving insert_slice the interface) is exactly R1's job to attack.

## 5. Reusable lesson captured (to memory)

**Trace the identity of a value at a line before building a thesis on it.** "The
stride is read at `:2311`" was true but vacuous until `:2311` was confirmed to
read the internal unit-stride candidate, not the external `[2,1]` store. This
misread recurred (an earlier advisory endorsed it without tracing
`candidateSlices`). Stored to long-term memory so the reflex is: any
"the data is already there, just propagate it" claim requires naming the exact
op that produces the value at that line.

## 6. Subagent reasoning files (index)

- A1 (rework) reasoning → `approach1_contract_rework_a1_reasoning.md`
- R1 (review) reasoning → embedded in `approach1_contract_plan_review_r1.md`
  (R1's brief required file:line-grounded reasoning per claim)
- A2 / R2 (if spawned) → each embeds a "Reasoning & Justification" section per
  the updated assignment template

## 7. Loop progress + my own verification checkpoint

- **A1 (rework): DONE.** Plan `approach1_contract_phase1_plan.md`; reasoning
  `approach1_contract_rework_a1_reasoning.md`. Chose mechanism (c): tile
  `tensor.insert_slice` as the anchor; stride source = its own `strides` attr.
  Call chain verified end-to-end; RED is a clean "not TilingInterface" at
  `LinalgTransformOps.cpp:3899-3900`.
- **R1 (review): DONE — verdict FIXABLE-WITH-SPECIFIC-CHANGES (conditional GO).**
  Review `approach1_contract_plan_review_r1.md` (7 findings). Headline BLOCKER:
  A1's `resultSizes = iterSizes*stride` is inverted.
- **My F1 verification (orchestrator, not delegated):** before spawning A2 I
  re-derived the insert_slice result-type semantics against
  `lib/Dialect/Tensor/IR/TensorOps.cpp:2885-2896` — `verifyInsertSliceOp` uses
  `ExtractSliceOp::inferResultType(dstType, staticSizes)` (**sizes only, no
  strides**), so `%tile.shape` must EQUAL `sizes`. **R1's F1 is correct**; my
  first-pass intuition (ceilDiv(sizes,strides)) was wrong — I had conflated
  MLIR's "sizes = source shape, strides space them" with a "sizes = span"
  convention. Correct formula: `resultSizes = iterSizes`. No-overlap contract
  (`TilingInterface.td:146-147`) is satisfied under the corrected convention.
  G1/G2 tensor guards confirmed OFF the initial-tiling path (irrelevant to G3/G4).
- **A2 (rework): DONE.** Edited plan v1→v2 (`approach1_contract_phase1_plan.md`,
  1045 lines) + per-finding changelog `approach1_contract_rework_a2_changes.md`.
  Fixed F1 (`resultSizes=iterSizes` everywhere), F2 (anchor sizes `[2,4]`),
  added §4a blast-radius containment, corrected the EXEC-lowering mis-citation to
  the real `ParallelInsertSliceOpInterface::bufferize` path, stated the
  `getDroppedDims` rule, PROMOTED numThreads EXEC to core G4b (justified + itself
  a genuine scatter), reworded §10 GO to IR-level-only, relabeled per-tile dest to
  region iter arg `%o0`.
- **Orchestrator scatter-vs-offset directive (decisive, my own verification):**
  I re-derived placement vs `TensorOps.cpp:2885-2896` — a size-1 strided dim has a
  VESTIGIAL stride (j=0 only → `offset+0*stride`, identical to unit-stride), so
  tile-size-1 proves only the strided *offset*, not a genuine *scatter*. I had A2
  make full-source sub-case A (genuine within-tile scatter) the PRIMARY G3/G4a
  gate and demote B to offset-placement with an explicit false-green NO-GO mode.
  A2 applied the same lens to numThreads G4b. This was the difference between a
  probe that proves the dispatch-relevant capability and one that proves an offset
  trick.
- **R2 (final review): DONE — CONVERGED (ready to execute).** Review
  `approach1_contract_plan_review_r2.md`. All six A2 fixes re-verified against
  source. **A2's hardest `[INFERENCE]` resolved POSITIVELY** by R2:
  `MemRefCopyOpLowering` (`MemRefToLLVM.cpp:1257-1278`) routes any strided dest to
  a generic element-wise `MemrefCopyFn` honoring strides — G4 is reachable (perf
  caveat: scalarized copy; not a compile/correctness wall). numThreads G4b math
  re-derived sound. Remaining items are MEDIUM/LOW execution risks, not A3
  blockers: (m1) §4a marker-gate names an impossible point — `getIterationDomain`
  returns `SmallVector<Range>` not `LogicalResult`; gate at
  `getTiledImplementation` instead (self-correcting at compile); (m2) §8 EXEC
  pipeline omits `-convert-memref-to-llvm` (the pass owning `MemRefCopyOpLowering`);
  (m3) G4a 1-iteration forall foldable → G3 FileCheck fragility (fallback =
  2-tile = G4b shape); (m4) §4a containment is best-effort, not hard.
- **A3 (polish, user option 1): DONE — execution-ready.** Fixed all four R2
  items in `approach1_contract_phase1_plan.md` (v3) + changelog
  `approach1_contract_rework_a3_changes.md`. m1: moved the marker-gate off
  `getIterationDomain` (returns `SmallVector<Range>`, can't fail) onto
  `getTiledImplementation`/`getResultTilePosition`; re-traced the bail path to
  confirm containment. m2: added `-expand-strided-metadata -finalize-memref-to-llvm`
  to the EXEC pipeline — **and corrected a stale cite of R2's own** (R2 named
  `-convert-memref-to-llvm`, which does not exist in this checkout; the real pass
  is `-finalize-memref-to-llvm`, `Passes.td:994-995`). m3: made G3 FileCheck
  fold-robust (match either op mnemonic). m4: honest best-effort-containment note.
- **Orchestrator decision (per advisory): NO R3.** R2 already CONVERGED; m1/m2
  are fail-fast (m1 = compile error at first build; m2 = "failed to legalize
  memref.copy" at first EXEC run) with zero silent-miscompile risk. A 7th agent
  round has near-zero marginal value; the plan's own GO criterion is an EXEC
  test, so the RED test + `mlir-opt` run outranks more planning.
- **Loop status:** CONVERGED + polished (A1→R1→A2→R2→A3). Plan v3 is
  **execution-ready**. All reasoning on disk: A1/A2/A3 reasoning+changelog files,
  R1/R2 reviews, this log. Next action is Stage 0 (build `mlir-opt`/`FileCheck`/
  `mlir-cpu-runner`) then the RED lit test — awaiting user go-ahead.
