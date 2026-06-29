// Note: this file is for patterns explicitly added only during the
// flow-specific canonicalization pass. Canonicalization patterns registered on
// flow dialect ops should be tested under the appropriate
// iree/compiler/Dialect/Flow/IR/test/*_folding.mlir file for the op category.

// RUN: iree-opt --iree-flow-canonicalize %s --split-input-file --mlir-print-local-scope | FileCheck %s

util.func public @fold_full_insert_into_extract(
    %source: tensor<8x?xf32>,
    %dest: tensor<10x?xf32>,
    %size: index) -> tensor<8x?xf32> {
  %extract = tensor.extract_slice %dest [1, 1] [8, %size] [1, 1] : tensor<10x?xf32> to tensor<8x?xf32>
  %insert = tensor.insert_slice %source into %extract [0, 0] [8, %size] [1, 1] : tensor<8x?xf32> into tensor<8x?xf32>
  util.return %insert : tensor<8x?xf32>
}

// CHECK-LABEL: util.func public @fold_full_insert_into_extract
//  CHECK-SAME:   %[[SOURCE:.+]]: tensor<8x?xf32>
//       CHECK:   util.return %[[SOURCE]]

// -----

util.func public @fold_full_insert_into_empty(
    %source: tensor<8x?xf32>,
    %size: index) -> tensor<8x?xf32> {
  %empty = tensor.empty(%size) : tensor<8x?xf32>
  %insert = tensor.insert_slice %source into %empty [0, 0] [8, %size] [1, 1] : tensor<8x?xf32> into tensor<8x?xf32>
  util.return %insert : tensor<8x?xf32>
}

// CHECK-LABEL: util.func public @fold_full_insert_into_empty
//  CHECK-SAME:   %[[SOURCE:.+]]: tensor<8x?xf32>
//       CHECK:   util.return %[[SOURCE]]

// -----

util.func public @dont_fold_not_full_insert_into_empty(
    %source: tensor<8x?xf32>,
    %size1: index, %size2: index) -> tensor<8x?xf32> {
  %empty = tensor.empty(%size1) : tensor<8x?xf32>
  %insert = tensor.insert_slice %source into %empty [0, 0] [8, %size2] [1, 1] : tensor<8x?xf32> into tensor<8x?xf32>
  util.return %insert : tensor<8x?xf32>
}

// CHECK-LABEL: util.func public @dont_fold_not_full_insert_into_empty
//       CHECK:   %[[INSERT:.+]] = tensor.insert_slice
//       CHECK:   util.return %[[INSERT]]

// -----

util.func public @dont_fold_not_full_static_insert_into_empty(
    %source: tensor<8x?xf32>,
    %size: index) -> tensor<10x?xf32> {
  %empty = tensor.empty(%size) : tensor<10x?xf32>
  %insert = tensor.insert_slice %source into %empty [0, 0] [8, %size] [1, 1] : tensor<8x?xf32> into tensor<10x?xf32>
  util.return %insert : tensor<10x?xf32>
}

// CHECK-LABEL: util.func public @dont_fold_not_full_static_insert_into_empty
//       CHECK:   %[[INSERT:.+]] = tensor.insert_slice
//       CHECK:   util.return %[[INSERT]]

// -----

util.func public @expand_affine(%arg0: index) -> index {
  %mul = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%arg0]
  util.return %mul : index
}

// CHECK-LABEL: util.func public @expand_affine
//  CHECK-SAME:   %[[ARG0:.+]]: index
//       CHECK:   %[[MUL:.+]] = arith.muli %[[ARG0]], %c4 overflow<nsw>
//       CHECK:   util.return %[[MUL]]

// -----

// `linalg.generic` that yields a single invariant scalar across its whole
// output (0 inputs, all-parallel, body a bare `linalg.yield %cst`) is
// equivalent to `linalg.fill` and is normalized to it. IREE codegen treats
// `linalg.fill` as a uniform fill (memset-friendly, zero-init detection); this
// is the spelling frontends emit when they materialize a constant broadcast as
// a generic (e.g. torch-mlir's `aten.fill.Tensor` with a 0-d value operand).
util.func public @fold_generic_yield_constant_to_fill(%size: index) -> tensor<8x?xf32> {
  %empty = tensor.empty(%size) : tensor<8x?xf32>
  %cst = arith.constant 0.0 : f32
  %generic = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} outs(%empty : tensor<8x?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
  } -> tensor<8x?xf32>
  util.return %generic : tensor<8x?xf32>
}

//      CHECK-LABEL: util.func public @fold_generic_yield_constant_to_fill
//            CHECK:   %[[FILL:.+]] = linalg.fill
//       CHECK-NOT:   linalg.generic
//            CHECK:   util.return %[[FILL]]

// -----

// Negative case: the yielded value varies per element (computed from the loop
// index via `linalg.index`), so the generic is genuinely non-uniform and must
// NOT be folded to a `linalg.fill`.
util.func public @dont_fold_generic_index_dependent(%size: index) -> tensor<8x?xf32> {
  %empty = tensor.empty(%size) : tensor<8x?xf32>
  %generic = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} outs(%empty : tensor<8x?xf32>) {
    ^bb0(%out: f32):
      %idx = linalg.index 0 : index
      %i = arith.index_cast %idx : index to i32
      %f = arith.sitofp %i : i32 to f32
      linalg.yield %f : f32
  } -> tensor<8x?xf32>
  util.return %generic : tensor<8x?xf32>
}

//      CHECK-LABEL: util.func public @dont_fold_generic_index_dependent
//            CHECK:   linalg.generic
//       CHECK-NOT:   linalg.fill
//            CHECK:   util.return

// -----

// FoldNestedInsertSlice: the core identity. Writing %src into a uniform fill of
// V at an inner [1,2] stride, then writing the result into an *equal* uniform
// fill of V at an outer [2,1] stride, composes to a single in-place write of
// %src at [2,2] directly into the base. Dynamic tensor types keep the inner and
// outer destinations type-equal (a genuinely strided static insert would be
// rank-reduced and bail); the intermediate fill and inner insert become dead
// and are DCE'd. Inner/outer sizes are intentionally mismatched so the upstream
// InsertSliceOfInsertSliceFolder (which needs matching sizes) stays out of the
// way and this IREE-specific pattern is provably the one that fires.
util.func public @fold_doubly_strided_uniform_fill(
    %src: tensor<?x?xf32>,
    %is0: index, %is1: index,
    %os0: index, %os1: index) -> tensor<?x?xf32> {
  %cst = arith.constant 0.0 : f32
  %base_e = tensor.empty(%os0, %os1) : tensor<?x?xf32>
  %base = linalg.fill ins(%cst : f32) outs(%base_e : tensor<?x?xf32>) -> tensor<?x?xf32>
  %inter_e = tensor.empty(%os0, %os1) : tensor<?x?xf32>
  %inter = linalg.fill ins(%cst : f32) outs(%inter_e : tensor<?x?xf32>) -> tensor<?x?xf32>
  %inner = tensor.insert_slice %src into %inter[0, 0] [%is0, %is1] [1, 2] : tensor<?x?xf32> into tensor<?x?xf32>
  %outer = tensor.insert_slice %inner into %base[0, 0] [%os0, %os1] [2, 1] : tensor<?x?xf32> into tensor<?x?xf32>
  util.return %outer : tensor<?x?xf32>
}
// CHECK-LABEL: util.func public @fold_doubly_strided_uniform_fill
//       CHECK: %[[FILL:.+]] = linalg.fill
//       CHECK: %[[RES:.+]] = tensor.insert_slice %{{.+}} into %[[FILL]][0, 0] [%{{.+}}, %{{.+}}] [2, 2]
//   CHECK-NOT: tensor.insert_slice
//       CHECK: util.return %[[RES]]

// -----

// Extract form of the same identity: the intermediate is an extract_slice of the
// base at the *same* slice as the outer insert (gated by isSameAs). The nested
// chain collapses to a single [2,2] insert straight into the base and the
// extract is eliminated entirely.
util.func public @fold_doubly_strided_extract_intermediate(
    %base: tensor<?x?xf32>,
    %src: tensor<?x?xf32>,
    %is0: index, %is1: index,
    %os0: index, %os1: index) -> tensor<?x?xf32> {
  %inter = tensor.extract_slice %base[0, 0] [%os0, %os1] [2, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  %inner = tensor.insert_slice %src into %inter[0, 0] [%is0, %is1] [1, 2] : tensor<?x?xf32> into tensor<?x?xf32>
  %outer = tensor.insert_slice %inner into %base[0, 0] [%os0, %os1] [2, 1] : tensor<?x?xf32> into tensor<?x?xf32>
  util.return %outer : tensor<?x?xf32>
}
// CHECK-LABEL: util.func public @fold_doubly_strided_extract_intermediate
//   CHECK-NOT: tensor.extract_slice
//       CHECK: %[[RES:.+]] = tensor.insert_slice %{{.+}} into %{{.+}}[0, 0] [%{{.+}}, %{{.+}}] [2, 2]
//   CHECK-NOT: tensor.insert_slice
//       CHECK: util.return %[[RES]]

// -----

// Static, non-zero offsets: the composed offset
// outerOffset + innerOffset * outerStride constant-folds. Here dim0 = 5 + 3*2
// = 11 and dim1 = 2 + 1*1 = 3, so the single insert lands at [11, 3] with no
// arith op materialized (contrast with the dynamic-offset case below).
util.func public @fold_doubly_strided_static_nonzero_offsets(
    %src: tensor<?x?xf32>,
    %is0: index, %is1: index,
    %os0: index, %os1: index) -> tensor<?x?xf32> {
  %cst = arith.constant 0.0 : f32
  %base_e = tensor.empty(%os0, %os1) : tensor<?x?xf32>
  %base = linalg.fill ins(%cst : f32) outs(%base_e : tensor<?x?xf32>) -> tensor<?x?xf32>
  %inter_e = tensor.empty(%os0, %os1) : tensor<?x?xf32>
  %inter = linalg.fill ins(%cst : f32) outs(%inter_e : tensor<?x?xf32>) -> tensor<?x?xf32>
  %inner = tensor.insert_slice %src into %inter[3, 1] [%is0, %is1] [1, 2] : tensor<?x?xf32> into tensor<?x?xf32>
  %outer = tensor.insert_slice %inner into %base[5, 2] [%os0, %os1] [2, 1] : tensor<?x?xf32> into tensor<?x?xf32>
  util.return %outer : tensor<?x?xf32>
}
// CHECK-LABEL: util.func public @fold_doubly_strided_static_nonzero_offsets
//       CHECK: %[[RES:.+]] = tensor.insert_slice %{{.+}} into %{{.+}}[11, 3] [%{{.+}}, %{{.+}}] [2, 2]
//   CHECK-NOT: tensor.insert_slice
//       CHECK: util.return %[[RES]]

// -----

// Dynamic offsets compose via affine.apply, which this pass's
// AffineApplyLowering then expands to arith. The per-dimension composed offset
// is outerOffset + innerOffset * outerStride: dim0 = %a0 + %b0*2 (muli + addi),
// dim1 = %a1 + %b1*1 (the *1 folds to a bare addi). Strides compose to [2,2].
// Static offsets (the cases above) fold all the way to integer constants.
util.func public @fold_doubly_strided_dynamic_offsets(
    %src: tensor<?x?xf32>,
    %is0: index, %is1: index,
    %os0: index, %os1: index,
    %a0: index, %a1: index,
    %b0: index, %b1: index) -> tensor<?x?xf32> {
  %cst = arith.constant 0.0 : f32
  %base_e = tensor.empty(%os0, %os1) : tensor<?x?xf32>
  %base = linalg.fill ins(%cst : f32) outs(%base_e : tensor<?x?xf32>) -> tensor<?x?xf32>
  %inter_e = tensor.empty(%os0, %os1) : tensor<?x?xf32>
  %inter = linalg.fill ins(%cst : f32) outs(%inter_e : tensor<?x?xf32>) -> tensor<?x?xf32>
  %inner = tensor.insert_slice %src into %inter[%b0, %b1] [%is0, %is1] [1, 2] : tensor<?x?xf32> into tensor<?x?xf32>
  %outer = tensor.insert_slice %inner into %base[%a0, %a1] [%os0, %os1] [2, 1] : tensor<?x?xf32> into tensor<?x?xf32>
  util.return %outer : tensor<?x?xf32>
}
// CHECK-LABEL: util.func public @fold_doubly_strided_dynamic_offsets
//       CHECK: %[[M:.+]] = arith.muli %{{.+}}, %{{.+}} overflow<nsw> : index
//       CHECK: %[[O0:.+]] = arith.addi %{{.+}}, %[[M]] : index
//       CHECK: %[[O1:.+]] = arith.addi %{{.+}}, %{{.+}} : index
//       CHECK: %[[RES:.+]] = tensor.insert_slice %{{.+}} into %{{.+}}[%[[O0]], %[[O1]]] [%{{.+}}, %{{.+}}] [2, 2]
//   CHECK-NOT: tensor.insert_slice
//       CHECK: util.return %[[RES]]

// -----

// Negative: the two fills use *different* scalar values (0.0 vs 1.0), so the
// intermediate is NOT value-equivalent to a strided subregion of the base. The
// value-equivalence gate fails and both inserts are kept. (areEqualScalarValues
// compares constant attributes, so distinct literals are required: two
// `arith.constant 0.0` would be equal and WOULD fold.)
util.func public @dont_fold_differently_filled_intermediate(
    %src: tensor<?x?xf32>,
    %is0: index, %is1: index,
    %os0: index, %os1: index) -> tensor<?x?xf32> {
  %c0 = arith.constant 0.0 : f32
  %c1 = arith.constant 1.0 : f32
  %base_e = tensor.empty(%os0, %os1) : tensor<?x?xf32>
  %base = linalg.fill ins(%c0 : f32) outs(%base_e : tensor<?x?xf32>) -> tensor<?x?xf32>
  %inter_e = tensor.empty(%os0, %os1) : tensor<?x?xf32>
  %inter = linalg.fill ins(%c1 : f32) outs(%inter_e : tensor<?x?xf32>) -> tensor<?x?xf32>
  %inner = tensor.insert_slice %src into %inter[0, 0] [%is0, %is1] [1, 2] : tensor<?x?xf32> into tensor<?x?xf32>
  %outer = tensor.insert_slice %inner into %base[0, 0] [%os0, %os1] [2, 1] : tensor<?x?xf32> into tensor<?x?xf32>
  util.return %outer : tensor<?x?xf32>
}
// CHECK-LABEL: util.func public @dont_fold_differently_filled_intermediate
//   CHECK-NOT: [2, 2]
//       CHECK: tensor.insert_slice %{{.+}}[0, 0] [%{{.+}}, %{{.+}}] [1, 2]
//       CHECK: tensor.insert_slice %{{.+}}[0, 0] [%{{.+}}, %{{.+}}] [2, 1]
//       CHECK: util.return

// -----

// Negative: the intermediate is a genuinely non-uniform compute (an
// element-index-dependent linalg.generic), so it is neither a uniform fill nor
// an extract of the base. The equivalence gate fails and both inserts are kept.
util.func public @dont_fold_nonuniform_intermediate(
    %src: tensor<?x?xf32>,
    %is0: index, %is1: index,
    %os0: index, %os1: index) -> tensor<?x?xf32> {
  %cst = arith.constant 0.0 : f32
  %base_e = tensor.empty(%os0, %os1) : tensor<?x?xf32>
  %base = linalg.fill ins(%cst : f32) outs(%base_e : tensor<?x?xf32>) -> tensor<?x?xf32>
  %inter_e = tensor.empty(%os0, %os1) : tensor<?x?xf32>
  %inter = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} outs(%inter_e : tensor<?x?xf32>) {
    ^bb0(%out: f32):
      %idx = linalg.index 0 : index
      %i = arith.index_cast %idx : index to i32
      %f = arith.sitofp %i : i32 to f32
      linalg.yield %f : f32
  } -> tensor<?x?xf32>
  %inner = tensor.insert_slice %src into %inter[0, 0] [%is0, %is1] [1, 2] : tensor<?x?xf32> into tensor<?x?xf32>
  %outer = tensor.insert_slice %inner into %base[0, 0] [%os0, %os1] [2, 1] : tensor<?x?xf32> into tensor<?x?xf32>
  util.return %outer : tensor<?x?xf32>
}
// CHECK-LABEL: util.func public @dont_fold_nonuniform_intermediate
//   CHECK-NOT: [2, 2]
//       CHECK: linalg.generic
//       CHECK: tensor.insert_slice %{{.+}}[0, 0] [%{{.+}}, %{{.+}}] [1, 2]
//       CHECK: tensor.insert_slice %{{.+}}[0, 0] [%{{.+}}, %{{.+}}] [2, 1]
//       CHECK: util.return

// -----

// Negative: a single (non-nested) insert_slice. The pattern requires the outer
// insert's source to itself be an insert_slice; it is not, so nothing fires and
// the insert is left unchanged.
util.func public @dont_fold_single_insert(
    %src: tensor<?x?xf32>,
    %os0: index, %os1: index) -> tensor<?x?xf32> {
  %cst = arith.constant 0.0 : f32
  %base_e = tensor.empty(%os0, %os1) : tensor<?x?xf32>
  %base = linalg.fill ins(%cst : f32) outs(%base_e : tensor<?x?xf32>) -> tensor<?x?xf32>
  %r = tensor.insert_slice %src into %base[0, 0] [%os0, %os1] [2, 1] : tensor<?x?xf32> into tensor<?x?xf32>
  util.return %r : tensor<?x?xf32>
}
// CHECK-LABEL: util.func public @dont_fold_single_insert
//   CHECK-NOT: [2, 2]
//       CHECK: %[[RES:.+]] = tensor.insert_slice %{{.+}}[0, 0] [%{{.+}}, %{{.+}}] [2, 1]
//   CHECK-NOT: tensor.insert_slice
//       CHECK: util.return %[[RES]]

// -----

// Negative: a rank-reduced intermediate. The inner insert's destination (1-D)
// does not have the same type as the outer insert's destination (2-D), so the
// pattern bails on the type check before attempting stride composition. (This
// is also why the positive cases use dynamic tensor types: any genuinely
// strided, equal-rank insert on static shapes would shrink the source type and
// trip this same bail.)
util.func public @dont_fold_rank_reduced_intermediate(
    %base: tensor<?x?xf32>,
    %src1d: tensor<?xf32>,
    %is: index, %os: index) -> tensor<?x?xf32> {
  %inter_e = tensor.empty(%is) : tensor<?xf32>
  %inter = tensor.insert_slice %src1d into %inter_e[0] [%is] [2] : tensor<?xf32> into tensor<?xf32>
  %outer = tensor.insert_slice %inter into %base[0, 0] [1, %os] [2, 1] : tensor<?xf32> into tensor<?x?xf32>
  util.return %outer : tensor<?x?xf32>
}
// CHECK-LABEL: util.func public @dont_fold_rank_reduced_intermediate
//   CHECK-NOT: [2, 2]
//       CHECK: tensor.insert_slice %{{.+}}[0] [%{{.+}}] [2] : tensor<?xf32> into tensor<?xf32>
//       CHECK: tensor.insert_slice %{{.+}}[0, 0] [1, %{{.+}}] [2, 1] : tensor<?xf32> into tensor<?x?xf32>
//       CHECK: util.return

// -----

// Negative: a dynamic (Value) stride. The composition forms
// outerStride * innerStride as a constant attribute, so a dynamic stride makes
// getConstantIntValue return nullopt and the pattern bails, leaving both
// inserts unchanged.
util.func public @dont_fold_dynamic_stride(
    %src: tensor<?x?xf32>,
    %is0: index, %is1: index,
    %os0: index, %os1: index,
    %dynstride: index) -> tensor<?x?xf32> {
  %cst = arith.constant 0.0 : f32
  %base_e = tensor.empty(%os0, %os1) : tensor<?x?xf32>
  %base = linalg.fill ins(%cst : f32) outs(%base_e : tensor<?x?xf32>) -> tensor<?x?xf32>
  %inter_e = tensor.empty(%os0, %os1) : tensor<?x?xf32>
  %inter = linalg.fill ins(%cst : f32) outs(%inter_e : tensor<?x?xf32>) -> tensor<?x?xf32>
  %inner = tensor.insert_slice %src into %inter[0, 0] [%is0, %is1] [1, %dynstride] : tensor<?x?xf32> into tensor<?x?xf32>
  %outer = tensor.insert_slice %inner into %base[0, 0] [%os0, %os1] [2, 1] : tensor<?x?xf32> into tensor<?x?xf32>
  util.return %outer : tensor<?x?xf32>
}
// CHECK-LABEL: util.func public @dont_fold_dynamic_stride
//   CHECK-NOT: [2, 2]
//       CHECK: tensor.insert_slice %{{.+}}[0, 0] [%{{.+}}, %{{.+}}] [1, %{{.+}}]
//       CHECK: tensor.insert_slice %{{.+}}[0, 0] [%{{.+}}, %{{.+}}] [2, 1]
//       CHECK: util.return
