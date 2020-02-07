//
// Created by gkarlos on 13-01-20.
//

#ifndef KERMA_STATIC_ANALYSIS_LLVMSTRINGUTILS_H
#define KERMA_STATIC_ANALYSIS_LLVMSTRINGUTILS_H

#include <llvm/IR/Value.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Support/raw_ostream.h>

#include <string>

namespace kerma {

/*
 * @brief Retrieve an std::string out of an llvm::Value
 */
std::string getStringFromLLVMValue( const llvm::Value *v);

/*
 * @brief Retrieve an std::string out of an llvm::Instruction
 */
std::string getStringFromLLVMInstr( const llvm::Instruction *v);

/*
 * @brief Retrieve an std::string for the source code loc
 * of the instruction in the form if "line:col".
 *
 * Returns "" when no location info can be found
 */
std::string getDbgLocString( const llvm::Instruction *I);

/*
 * @brief Trim leading whitespace (in-place)
 */
std::string& ltrim(std::string &s);
std::string& ltrim(std::string &&s);
}

#endif // KERMA_STATIC_ANALYSIS_LLVMSTRINGUTILS_H
