set(LIBS
  # Local dialects.
  IREELinalgTransformDialect
  IREELinalgTransformDialectPasses
  # Core dialects.
  MLIRAffineDialect
  MLIRArithDialect
  MLIRAsyncDialect
  MLIRBufferizationTransformOps
  MLIRControlFlowDialect
  MLIRDialect
  MLIRFuncDialect
  MLIRFuncInlinerExtension
  MLIRIndexToLLVM
  MLIRLLVMDialect
  MLIRLLVMIRTransforms
  MLIRLinalgDialect
  MLIRLinalgTransformOps
  MLIRMemRefDialect
  MLIRMemRefTransforms
  MLIROptLib
  MLIRPDLDialect
  MLIRPDLInterpDialect
  MLIRSCFDialect
  MLIRSCFTransformOps
  MLIRSCFTransforms
  MLIRTensorDialect
  MLIRTensorTransformOps
  MLIRTransforms
  MLIRTransformPDLExtension
  MLIRVectorTransformOps
)

add_llvm_executable(iree-dialects-opt
  iree-dialects-opt.cpp

  DEPENDS
  ${LIBS}
)
target_link_libraries(iree-dialects-opt PRIVATE ${LIBS})
