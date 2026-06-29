// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::Flow {

#define GEN_PASS_DEF_CANONICALIZEPASS
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h.inc"

namespace {

static std::optional<SmallVector<OpFoldResult>> getDefiningMixedSizes(Value v) {
  if (auto empty = v.getDefiningOp<tensor::EmptyOp>()) {
    return empty.getMixedSizes();
  } else if (auto extract = v.getDefiningOp<tensor::ExtractSliceOp>()) {
    // TODO: Support rank reducing cases.
    if (extract.getSourceType().getRank() !=
        extract.getResultType().getRank()) {
      return {};
    }
    return extract.getMixedSizes();
  }
  return {};
}

struct FoldFullInsertSlice : OpRewritePattern<tensor::InsertSliceOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(tensor::InsertSliceOp insertSliceOp,
                                PatternRewriter &rewriter) const override {
    if (!insertSliceOp.hasUnitStride() || !insertSliceOp.hasZeroOffset()) {
      return rewriter.notifyMatchFailure(insertSliceOp,
                                         "non-unit stride or non-zero offset.");
    }

    RankedTensorType sourceType = insertSliceOp.getSourceType();
    RankedTensorType resultType = insertSliceOp.getResultType();
    if (sourceType != resultType) {
      return rewriter.notifyMatchFailure(
          insertSliceOp,
          "unimplemented: Cast-like or reshape-like insert ops.");
    }

    std::optional<SmallVector<OpFoldResult>> mixedSizes =
        getDefiningMixedSizes(insertSliceOp.getDest());
    if (!mixedSizes) {
      return rewriter.notifyMatchFailure(
          insertSliceOp, "Could not find producer with list of tensor sizes.");
    }

    for (auto [insertSize, destSize] :
         llvm::zip_equal(insertSliceOp.getMixedSizes(), mixedSizes.value())) {
      if (isa<Value>(insertSize) || isa<Value>(destSize)) {
        if (insertSize != destSize) {
          return rewriter.notifyMatchFailure(insertSliceOp,
                                             "dynamic size mismatch");
        }
        continue;
      }

      // `getMixedSizes` for different ops returns different attribute types
      // (`index` or `i64`) so we compare the values of the ints directly here.
      int64_t staticInsertSize = getConstantIntValue(insertSize).value();
      int64_t staticDestSize = getConstantIntValue(insertSize).value();
      if (staticInsertSize != staticDestSize) {
        return rewriter.notifyMatchFailure(insertSliceOp,
                                           "static size mismatch");
      }
    }

    rewriter.replaceOp(insertSliceOp, insertSliceOp.getSource());
    return success();
  }
};

/// Convert an "affine.apply" operation into a sequence of arith ops.
class AffineApplyLowering : public OpRewritePattern<affine::AffineApplyOp> {
public:
  using Base::Base;

  LogicalResult matchAndRewrite(affine::AffineApplyOp op,
                                PatternRewriter &rewriter) const override {
    auto maybeExpandedMap =
        affine::expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(),
                                llvm::to_vector<8>(op.getOperands()));
    if (!maybeExpandedMap) {
      return failure();
    }
    rewriter.replaceOp(op, *maybeExpandedMap);
    return success();
  }
};

/// Return the scalar value of a uniformly-filled tensor (linalg.fill), or
/// nullopt if `v` is not a uniform-fill tensor.
static std::optional<Value> getUniformFillValue(Value v) {
  // linalg.fill's operand 0 is the fill value; the result is uniform.
  if (auto fillOp = v.getDefiningOp<linalg::FillOp>()) {
    return fillOp->getOperand(0);
  }
  return std::nullopt;
}

/// Return true if two scalar values are known to be equal.
static bool areEqualScalarValues(Value a, Value b) {
  if (a == b) {
    return true;
  }
  Attribute attrA, attrB;
  if (matchPattern(a, m_Constant(&attrA)) &&
      matchPattern(b, m_Constant(&attrB))) {
    return attrA == attrB;
  }
  return false;
}

/// Compose `outerOffset + innerOffset * outerStride` as an OpFoldResult.
static OpFoldResult composeInsertSliceOffset(OpBuilder &builder, Location loc,
                                             OpFoldResult outerOffset,
                                             OpFoldResult innerOffset,
                                             int64_t outerStride) {
  AffineExpr dOuter, dInner;
  bindDims(builder.getContext(), dOuter, dInner);
  AffineMap map = AffineMap::get(
      /*dimCount=*/2, /*symbolCount=*/0,
      dOuter + dInner * builder.getAffineConstantExpr(outerStride));
  return affine::makeComposedFoldedAffineApply(builder, loc, map,
                                               {outerOffset, innerOffset});
}

/// Fold a nested insert_slice chain where the inner write's destination is
/// value-equivalent to a strided subregion of the outer write's destination:
///
///   %inter = <uniform fill of V>   OR   %inter = extract_slice %base @ outer
///   %inner = tensor.insert_slice %src into %inter @ innerSlice
///   %outer = tensor.insert_slice %inner into %base @ outerSlice
///  ==>
///   %new = tensor.insert_slice %src into %base @ composedSlice
///
/// where, per dimension, composedSlice.offset = outerSlice.offset +
/// innerSlice.offset * outerSlice.stride, composedSlice.size = innerSlice.size,
/// and composedSlice.stride = outerSlice.stride * innerSlice.stride. Valid
/// whenever %inter equals extract_slice(%base) at outerSlice by value: when it
/// is literally that extract_slice, or when both %inter and %base are uniform
/// fills of the same value. This collapses two axis-disjoint strided writes
/// into a single in-place multi-strided write, avoiding a transposed
/// read-modify-write dispatch whose output store later cannot be distributed by
/// workgroup tiling.
struct FoldNestedInsertSlice : OpRewritePattern<tensor::InsertSliceOp> {
  using OpRewritePattern<tensor::InsertSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::InsertSliceOp outer,
                                PatternRewriter &rewriter) const override {
    auto inner = outer.getSource().getDefiningOp<tensor::InsertSliceOp>();
    if (!inner) {
      return failure();
    }

    Value base = outer.getDest();
    Value inter = inner.getDest();
    if (inter.getType() != base.getType()) {
      return failure(); // skip rank-reduced cases
    }

    // `inter` must be value-equivalent to extract_slice(%base) at `outer`'s
    // slice: either literally that extract_slice, or a uniform fill of the same
    // value as `base`'s uniform fill.
    auto isSame = [](OpFoldResult a, OpFoldResult b) { return a == b; };
    bool equivalent = false;
    if (auto extractOp = inter.getDefiningOp<tensor::ExtractSliceOp>()) {
      if (extractOp.getSource() == base && extractOp.isSameAs(outer, isSame)) {
        equivalent = true;
      }
    }
    if (!equivalent) {
      std::optional<Value> interVal = getUniformFillValue(inter);
      std::optional<Value> baseVal = getUniformFillValue(base);
      if (interVal && baseVal && areEqualScalarValues(*interVal, *baseVal)) {
        equivalent = true;
      }
    }
    if (!equivalent) {
      return failure();
    }

    SmallVector<OpFoldResult> outerOffsets = outer.getMixedOffsets();
    SmallVector<OpFoldResult> outerStrides = outer.getMixedStrides();
    SmallVector<OpFoldResult> innerOffsets = inner.getMixedOffsets();
    SmallVector<OpFoldResult> innerSizes = inner.getMixedSizes();
    SmallVector<OpFoldResult> innerStrides = inner.getMixedStrides();

    unsigned rank = outerOffsets.size();
    if (innerOffsets.size() != rank) {
      return failure();
    }

    Location loc = outer.getLoc();
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(outer);

    SmallVector<OpFoldResult> newOffsets;
    SmallVector<OpFoldResult> newStrides;
    newOffsets.reserve(rank);
    newStrides.reserve(rank);
    for (unsigned d = 0; d < rank; ++d) {
      std::optional<int64_t> outerStride = getConstantIntValue(outerStrides[d]);
      std::optional<int64_t> innerStride = getConstantIntValue(innerStrides[d]);
      if (!outerStride || !innerStride) {
        return failure(); // require static strides to compose
      }
      newOffsets.push_back(composeInsertSliceOffset(
          rewriter, loc, outerOffsets[d], innerOffsets[d], *outerStride));
      newStrides.push_back(rewriter.getIndexAttr(*outerStride * *innerStride));
    }

    rewriter.replaceOpWithNewOp<tensor::InsertSliceOp>(
        outer, inner.getSource(), base, newOffsets, innerSizes, newStrides);
    return success();
  }
};

/// Canonicalize operations in nested regions.
struct CanonicalizePass : impl::CanonicalizePassBase<CanonicalizePass> {
  using IREE::Flow::impl::CanonicalizePassBase<
      CanonicalizePass>::CanonicalizePassBase;
  /// Initialize the canonicalizer by building the set of patterns used during
  /// execution.
  LogicalResult initialize(MLIRContext *context) override {
    // Inherit the same config defaults from the upstream canonicalizer pass.
    config.setUseTopDownTraversal().setRegionSimplificationLevel(
        mlir::GreedySimplifyRegionLevel::Normal);

    RewritePatternSet owningPatterns(context);
    for (auto *dialect : context->getLoadedDialects()) {
      dialect->getCanonicalizationPatterns(owningPatterns);
    }
    for (RegisteredOperationName op : context->getRegisteredOperations()) {
      op.getCanonicalizationPatterns(owningPatterns, context);
    }

    // Pull in some borderline/downstream canonicalizations for the Flow
    // compilation phase.
    tensor::populateMergeConsecutiveInsertExtractSlicePatterns(owningPatterns);
    owningPatterns.add<FoldFullInsertSlice>(context);
    owningPatterns.add<AffineApplyLowering>(context);
    owningPatterns.add<FoldNestedInsertSlice>(context);

    patterns =
        std::make_shared<FrozenRewritePatternSet>(std::move(owningPatterns));
    return success();
  }
  void runOnOperation() override {
    // Canonicalization is best-effort. Non-convergence is not a pass failure.
    config.enableConstantCSE(cseConstants);
    LogicalResult didConverge =
        applyPatternsGreedily(getOperation(), *patterns, config);
    if (this->testConvergence && failed(didConverge)) {
      getOperation()->emitError("Canonicalizer failed to converge");
      return signalPassFailure();
    }
  }
  GreedyRewriteConfig config;
  std::shared_ptr<const FrozenRewritePatternSet> patterns;
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Flow
