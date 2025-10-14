// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Stream/Analysis/Affinity.h"
#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-stream-clone-to-consumers"

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_CLONETOCONSUMERSPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// --iree-stream-clone-to-consumers
//===----------------------------------------------------------------------===//

// Returns true if the given |op| can be cloned as part of this pass.
static bool canCloneOp(Operation *op) {
  if (!op) {
    return false;
  } else if (auto streamableOp =
                 dyn_cast<IREE::Stream::StreamableOpInterface>(op)) {
    return streamableOp.preferCloneToConsumers();
  } else if (mlir::isPure(op)) {
    return true;
  }
  return false;
}

// TODO(benvanik): swap this with a full analysis to find values that are on
// edges that should be cloned. For example, a solver given
// `A -> B -> C -> device0|device1 -> D` could mark A, B, and C as needing
// clones for device 0 and 1. If ops that consume values but are still cloneable
// are added we may need that to clone entire trees in one shot instead of
// needing the fixed-point iteration. It would also let us clone across branch
// and function boundaries: this simple local analysis only works in a single
// basic block.
static bool tryCloneToConsumersInRegion(Operation *op, Region &region,
                                        AffinityAnalysis &analysis) {
  [[maybe_unused]] std::unique_ptr<AsmState> asmState;
  LLVM_DEBUG(asmState = std::make_unique<AsmState>(op));

  bool didChange = false;
  SmallVector<IREE::Stream::AffinityAttr> affinities; // cached, cleared in for
  // Map from (defining_op -> (target_affinity -> cloned_op))
  // This persists across all consumers to enable clone reuse.
  DenseMap<Operation *, DenseMap<IREE::Stream::AffinityAttr, Operation *>>
      clonedOpsByAffinity;
  for (auto &block : region.getBlocks()) {
    // Note that we walk backwards so that we clone for later ops in the block
    // than for earlier ones to preserve IR order. This is not required for
    // correctness but does really help debugging.
    for (auto &op : llvm::reverse(block.getOperations())) {
      // Determine the target affinity for this consumer operation.
      // For operations like flow.tensor.transfer that have an explicit target,
      // we use that. Otherwise we try to infer the execution affinity.
      IREE::Stream::AffinityAttr targetAffinity;
      
      if (auto transferOp = dyn_cast<IREE::Flow::TensorTransferOp>(&op)) {
        // For transfer operations, use the target device they're transferring to
        targetAffinity = dyn_cast_or_null<IREE::Stream::AffinityAttr>(
            transferOp.getTarget());
      } else {
        // For other operations, try to infer their execution affinity
        SmallVector<IREE::Stream::AffinityAttr> consumerAffinities;
        analysis.tryInferExecutionAffinity(&op, consumerAffinities);
        if (consumerAffinities.size() == 1) {
          targetAffinity = consumerAffinities.front();
        }
      }
      for (auto &operand : op.getOpOperands()) {
        // This simple analysis is block local and is not be able to look across
        // branches or function calls.
        auto *definingOp = operand.get().getDefiningOp();
        if (!canCloneOp(definingOp)) {
          continue;
        }

        // If we already cloned the defining op for this target affinity, we can
        // reuse it across multiple consumers with the same affinity.
        auto result = cast<OpResult>(operand.get());
        
        // Check if we have a clone for this (definingOp, targetAffinity) pair.
        Operation *clonedOp = nullptr;
        if (targetAffinity) {
          auto defOpIt = clonedOpsByAffinity.find(definingOp);
          if (defOpIt != clonedOpsByAffinity.end()) {
            auto affinityIt = defOpIt->second.find(targetAffinity);
            if (affinityIt != defOpIt->second.end()) {
              clonedOp = affinityIt->second;
            }
          }
        }
        
        if (clonedOp) {
          LLVM_DEBUG({
            llvm::dbgs()
                << "[CloneToConsumers] * reusing clone for affinity ";
            targetAffinity.print(llvm::dbgs());
            llvm::dbgs() << " of operand ";
            result.printAsOperand(llvm::dbgs(), *asmState);
            llvm::dbgs() << " source: ";
            definingOp->print(llvm::dbgs(), *asmState);
            llvm::dbgs() << "\n";
          });
          operand.set(clonedOp->getResult(result.getResultNumber()));
          didChange = true;
          continue;
        }

        // Get the affinities the operand is potentially produced for.
        // This will fail if analysis failed or may return the default affinity.
        affinities.clear();
        analysis.tryLookupResourceAffinity(result, affinities);

        // Clone the producer of the operand if it has multiple affinities and
        // replace our use with it.
        if (affinities.size() > 1) {
          LLVM_DEBUG({
            llvm::dbgs() << "[CloneToConsumers] ! result ";
            result.printAsOperand(llvm::dbgs(), *asmState);
            llvm::dbgs() << " has multiple affinities: [";
            llvm::interleaveComma(affinities, llvm::dbgs());
            llvm::dbgs() << "]\n";
          });

          // If the op only has a single use (us) we can skip the clone. This
          // arises in cases where we've had to clone for other users before
          // reaching this op.
          if (llvm::hasSingleElement(definingOp->getUsers())) {
            LLVM_DEBUG({
              llvm::dbgs() << "[CloneToConsumers] ~ result op has one user "
                              "(us), not cloning: ";
              result.printAsOperand(llvm::dbgs(), *asmState);
              llvm::dbgs() << "\n";
            });
            continue;
          }
          
          // Check if all remaining users share the same target affinity as us.
          // If so, we don't need to clone since they'll all use the same value.
          if (targetAffinity) {
            bool allUsersShareAffinity = true;
            for (auto *user : definingOp->getUsers()) {
              IREE::Stream::AffinityAttr userAffinity;
              if (auto transferOp = dyn_cast<IREE::Flow::TensorTransferOp>(user)) {
                userAffinity = dyn_cast_or_null<IREE::Stream::AffinityAttr>(
                    transferOp.getTarget());
              } else {
                SmallVector<IREE::Stream::AffinityAttr> userAffinities;
                analysis.tryInferExecutionAffinity(user, userAffinities);
                if (userAffinities.size() == 1) {
                  userAffinity = userAffinities.front();
                }
              }
              if (userAffinity != targetAffinity) {
                allUsersShareAffinity = false;
                break;
              }
            }
            if (allUsersShareAffinity) {
              LLVM_DEBUG({
                llvm::dbgs() << "[CloneToConsumers] ~ all remaining users share "
                                "affinity, not cloning: ";
                result.printAsOperand(llvm::dbgs(), *asmState);
                llvm::dbgs() << "\n";
              });
              continue;
            }
          }

          LLVM_DEBUG({
            llvm::dbgs() << "[CloneToConsumers] + cloning operand ";
            result.printAsOperand(llvm::dbgs(), *asmState);
            llvm::dbgs() << " source: ";
            definingOp->print(llvm::dbgs(), *asmState);
            if (targetAffinity) {
              llvm::dbgs() << " for target affinity ";
              targetAffinity.print(llvm::dbgs());
            } else {
              llvm::dbgs() << " for unknown affinity";
            }
            llvm::dbgs() << "\n";
          });
          
          // Insert the clone right after the defining op to ensure it dominates
          // ALL consumers (including earlier ones we haven't processed yet when
          // walking backwards).
          OpBuilder builder(definingOp->getBlock(),
                            std::next(Block::iterator(definingOp)));
          auto *newClonedOp = builder.clone(*definingOp);
          
          // Store the clone indexed by target affinity for potential reuse.
          if (targetAffinity) {
            clonedOpsByAffinity[definingOp][targetAffinity] = newClonedOp;
          }
          
          operand.set(newClonedOp->getResult(result.getResultNumber()));

          didChange = true;
          continue;
        }
      }

      // Note: Cleanup of unused defining ops is now handled at the block level
      // after processing all operations, since clones are reused across
      // multiple consumers.
    }
    
    // Cleanup: erase original defining ops that are no longer used after all
    // operations in the block have been processed.
    for (auto &[definingOp, affinityMap] : clonedOpsByAffinity) {
      if (definingOp->use_empty()) {
        LLVM_DEBUG({
          llvm::dbgs() << "[CloneToConsumers] - erasing unused defining op: ";
          definingOp->print(llvm::dbgs(), *asmState);
          llvm::dbgs() << "\n";
        });
        definingOp->erase();
      }
    }
    // Clear the map for the next block to avoid holding stale pointers.
    clonedOpsByAffinity.clear();
  }
  return didChange;
}

// Clones ops that request cloning to consumers when their affinity is
// ambiguous.
struct CloneToConsumersPass
    : public IREE::Stream::impl::CloneToConsumersPassBase<
          CloneToConsumersPass> {
  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();
    if (moduleOp.getBody()->empty()) {
      return;
    }

    // NOTE: we currently run this only once because all current inputs only
    // need that. If we end up with more complex programs that have transfers
    // that break analysis we may need multiple runs.
    unsigned maxIterationCount = 32;
    LLVM_DEBUG(llvm::dbgs()
               << "[CloneToConsumers] beginning for maxIterationCount="
               << maxIterationCount << "\n");

    // Try analyzing the program and cloning operations until all are used on
    // a single affinity.
    unsigned iterationCount = 0;
    for (; iterationCount < maxIterationCount; ++iterationCount) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[CloneToConsumers] iteration " << iterationCount << "\n");

      // Perform whole-program analysis.
      // TODO(benvanik): reuse allocator across iterations.
      AffinityAnalysis analysis(moduleOp);
      if (failed(analysis.run())) {
        moduleOp.emitError() << "failed to solve for affinity analysis";
        return signalPassFailure();
      }

      // Apply analysis by cloning all ops we can with ambiguous affinities.
      // If we can't clone any we'll consider the iteration complete and exit.
      bool didChange = false;
      for (auto funcOp : moduleOp.getOps<CallableOpInterface>()) {
        bool funcDidChange = false;
        if (auto *region = funcOp.getCallableRegion()) {
          funcDidChange =
              tryCloneToConsumersInRegion(funcOp, *region, analysis);
        }
        didChange |= funcDidChange;
      }
      if (!didChange) {
        break;
      }
    }
    if (iterationCount == maxIterationCount) {
      // If you find yourself hitting this we can evaluate increasing the
      // iteration count (if it would eventually converge) or whether we allow
      // this to happen without remarking. For now all our programs converge in
      // just one or two iterations and this needs to be tuned with more complex
      // control flow.
      moduleOp.emitRemark()
          << "clone to consumers pass failed to reach a fixed point after "
          << maxIterationCount
          << " iterations; ambiguous affinity may be present";
      return;
    }

    LLVM_DEBUG(llvm::dbgs()
               << "[CloneToConsumers] completed after " << iterationCount << "/"
               << maxIterationCount << " iterations\n");
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Stream
