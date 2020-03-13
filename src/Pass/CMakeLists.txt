set(KermaPasses DetectKernels)

set(_Pass_LLVM_SRC "")

foreach(pass ${KermaPasses})
    add_subdirectory(${pass})
    list(APPEND _Pass_LLVM_SRC ${${pass}_LLVM_SRC})
endforeach(pass)

set(Pass_LLVM_SRC ${_Pass_LLVM_SRC} PARENT_SCOPE)