//
// Created by gkarlos on 1/5/20.
//

#ifndef KERMA_STATIC_ANALYSIS_PASSES_SUPPORT_H
#define KERMA_STATIC_ANALYSIS_PASSES_SUPPORT_H

#include <kerma/cuda/CudaSupport.h>

#include <llvm/IR/Module.h>
#include <llvm/Demangle/Demangle.h>

#include <string>

namespace kerma
{

/*
 * @brief Demangle a function name.
 *
 * @param [in] f Pointer to a function
 *
 * @details The LLVM demangler returns a function rather than just the
 *          function name. To retrieve the demangled function name without
 *          the arguments use #demangleFnWithoutAgs
 */
std::string demangleFn(llvm::Function *f);

/*
 * @brief Demangle a function name.
 *
 * @param [in] f Pointer to a function
 *
 * @details Strips the function args from the demangled named returned by the
 *          LLVM demangler (i.e. "(", ")" and anything in between. To retrieve
 *          the demangled name with the arguments use #demangleFn
 */
std::string demangleFnWithoutArgs(llvm::Function *f);

/*
 * @brief Check if a Module is a Device Side LLVM IR Module
 * @param [in] module An LLVM IR Module
 */
bool isDeviceModule(llvm::Module& module);

/*
 * @brief Check if a Module is a Host Side LLVM IR Module
 * @param [in] module An LLVM IR Module
 */
bool isHostModule(llvm::Module& module);

/*
 * @brief Retrieve the side an LLVM IR is relevant for (host, device)
 * @param [in] module An LLVM IR Module
 */
cuda::CudaSide getIRModuleSide(llvm::Module &module);

} // NAMESPACE kerma

#endif // KERMA_STATIC_ANALYSIS_PASSES_SUPPORT_H
