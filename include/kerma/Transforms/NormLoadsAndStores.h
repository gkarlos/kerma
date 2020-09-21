#ifndef KERMA_TRANSFORMS_NORMALIZE_LOADS_AND_STORES
#define KERMA_TRANSFORMS_NORMALIZE_LOADS_AND_STORES

#include <llvm/Pass.h>

namespace kerma {

/// This pass replaces constant pointers arguments in
/// loads and stores by a GEP instruction
///
/// Example:
///   define dso_local void @f(i32* %0) {
///      %1 = load i32, i32 *0;
///  }
///
///  ==>
//
///  define dso_local void @f(i32* %0) {
///      %1 = getelementptr inbounds i32, i32 *0, i64 0
///      %2 = load i32, i32 *1;
///  }
class NormLoadsAndStoresPass : public llvm::FunctionPass {
public:
  static char ID;
  NormLoadsAndStoresPass();
  bool runOnFunction(llvm::Function &F) override;
};

} // namespace kerma

#endif // KERMA_TRANSFORMS_NORMALIZE_LOADS_AND_STORES