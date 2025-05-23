// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_DIALECT_IREECODEGEN_DIALECT
#define IREE_CODEGEN_DIALECT_IREECODEGEN_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// IREE Codegen dialect
//===----------------------------------------------------------------------===//

def IREECodegen_Dialect : Dialect {
  let name = "iree_codegen";
  let cppNamespace = "::mlir::iree_compiler::IREE::Codegen";

  let summary = [{
    A dialect for common functionality used by IREE code generation.
  }];
  let description = [{
    This dialect is primarily meant to hold attributes that carry the
    state of the compilation when lowered to scalar code for an
    architecture. Typically, a backend starts by analyzing the entry
    point functions within the `hal.executable.variant` and deciding
    which compilation pipeline to chose. During this, even the values
    for parameters such as tile sizes, etc. are also decided. The rest
    of the compilation flow does not make any heuristic decisions,
    rather just looks at the values of the decision specified in
    attributes that belong to this dialect. This allows an external
    search to easily override the heuristics that are hard-coded
    within a backend.
  }];
  let extraClassDeclaration = [{
    void initializeCodegenAttrs();

    /// Returns the loaded module for the `libraryPath` transform dialect
    /// library MLIR input. If the module has already been loaded in the
    /// past, returns the memoized module without parsing the file again.
    /// This function is thead-safe.
    FailureOr<::mlir::ModuleOp>
    getOrLoadTransformLibraryModule(StringRef libraryPath);

    /// Returns the parsed module for the `libraryPath` transform dialect
    /// library MLIR input `libraryMLIRSource`. If the module has already
    /// been parsed in the past, returns the memoized module without parsing
    /// the source again. Note that this function does not access
    /// `libraryPath` and only uses it as the key associated with the
    /// matching parsed module.
    /// This function is thead-safe.
    FailureOr<::mlir::ModuleOp>
    getOrParseTransformLibraryModule(StringRef libraryPath,
                                     StringRef libraryMLIRSource);

    private:

    /// Map containing modules containing symbols, e.g. named sequences, that
    /// will be executed by the interpreter when used. This is a reflection of the
    /// library module storage upstream on the transform dialect, but instead we
    /// manage it here to ensure all required dialects are registered, and so that
    /// we can handle the loading/caching ourselves.
    ::llvm::StringMap<::mlir::OwningOpRef<::mlir::ModuleOp>> libraryModules;

    /// Lock to control the updating of the library modules such that we only load
    /// the module once and can reuse it across all invocations.
    std::mutex libraryMutex;
  }];
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let hasOperationAttrVerify = 1;
}

def AnyRankedTensorOrMemRefType : AnyTypeOf<[AnyRankedTensor, AnyMemRef]>;

class RankedTensorOrMemRefType<list<Type> allowedTypes, list<int> ranks>
    : AnyTypeOf<[TensorRankOf<allowedTypes, ranks>,
        MemRefRankOf<allowedTypes, ranks>]>;

#endif // IREE_CODEGEN_DIALECT_IREECODEGEN_DIALECT
