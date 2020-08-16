#ifndef KERMA_PASS_MATERIALIZE_GRID_H
#define KERMA_PASS_MATERIALIZE_GRID_H

#include "llvm/Pass.h"
namespace kerma {

class MaterializeGridPass : public llvm::ModulePass {

};

std::unique_ptr<MaterializeGridPass> createMaterializeGridPass();

}

#endif