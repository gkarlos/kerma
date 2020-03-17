set(KermaPasses DetectKernels)

set(Pass_LLVM_SRC "")
set(Pass_LLVM_COMP "")

foreach(pass ${KermaPasses})
    add_subdirectory(${pass})
    list(APPEND Pass_LLVM_SRC ${${pass}_LLVM_Sources})
    list(APPEND Pass_LLVM_COMP ${${pass}_LLVM_Components})
endforeach(pass)

set(Pass_LLVM_Sources ${Pass_LLVM_SRC} PARENT_SCOPE)
set(Pass_LLVM_Components ${Pass_LLVM_Components} PARENT_SCOPE)