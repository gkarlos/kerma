#ifndef KERMA_PASS_DETECT_KERNEL_LAUNCHES_H
#define KERMA_PASS_DETECT_KERNEL_LAUNCHES_H

#include "llvm/Pass.h"

/*
 * This Pass identifies Cuda kernel launch sites and launch configurations
 */
class DetectKernelLaunchesPass : llvm::ModulePass
{

};


#endif /* KERMA_PASS_DETECT_KERNEL_LAUNCHES_H */