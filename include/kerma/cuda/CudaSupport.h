//
// Created by gkarlos on 1/3/20.
//

#ifndef KERMA_STATIC_ANALYSIS_CUDA_SUPPORT_H
#define KERMA_STATIC_ANALYSIS_CUDA_SUPPORT_H

#include <string>

namespace kerma
{
namespace cuda
{

enum class CudaSide {
  Unknown,
  HOST,
  DEVICE
};

/**
 * Cuda Compute Capabilities
 */
enum class Compute {
  Unknown,
  CC30, CC32,        // Kepler
  CC35,
  CC37,
  CC50, CC52, CC53,  // Maxwell
  CC60, CC61, CC62,  // Pascal
  CC70, CC72,        // Volta
  CC75               // Turing
};

/**
 * Nvidia GPU Architectures
 */
enum class Arch {
  Unknown,
  SM30, SM32,        // Kepler
  SM35,
  SM50, SM52, SM53,  // Maxwell
  SM60, SM61, SM62,  // Pascal
  SM70, SM72,        // Volta
  SM75               // Turing
};

enum class AddressSpace {
  Unknown,
  GLOBAL,
  SHARED,
  CONSTANT,
  TEXTURE,
};

enum class BuiltIn {
  Unknown,
  GridDim,
  GridDimX,
  GridDimY,
  GridDimZ,
  BlockDim,
  BlockDimX,
  BlockDimY,
  BlockDimZ,
  BlockIdx,
  BlockIdxX,
  BlockIdxY,
  BlockIdxZ,
  ThreadIdx,
  ThreadIdxX,
  ThreadIdxY,
  ThreadIdxZ
};

/*
 * @brief Get a string value for a Cuda Compute Capability
 *
 * @param [in] c A Cuda Compute Capability
 * @param [in] full Get a string with a full description
 */
std::string cudaComputeToString(Compute c, bool full=false);

/*
 * @brief Get a string value of an Nvidia GPU Architecture
 *
 * @param [in] arch An Nvidia Architecture
 */
std::string cudaArchToString(Arch arch);
/*
 * @brief Get the name of an Nvidia GPU Architecture
 *
 * @param [in] arch An Nvidia architecture
 */
std::string cudaArchName(Arch arch);

/*
 * @brief Get the name of the GPU Architecture from a Compute Capability
 *
 * @param [in] cc A Compute Capability
 */
std::string cudaArchName(Compute cc);

/*
 * @brief Get an Nvidia Arch from a string
 *
 * @param [in] archName the name of the Nvidia Arch e.g sm_30, sm_52 etc
 */
Arch archFromString(const std::string& archName);

/*
 * @brief Get a string version of CudaSide
 *
 * @param [in] side a cuda side. e.g CudaSide::HOST
 */
std::string cudaSideToString(CudaSide side);

} // NAMESPACE cuda
} // NAMESPACE kerma

#endif // KERMA_STATIC_ANALYSIS_CUDA_SUPPORT_H
