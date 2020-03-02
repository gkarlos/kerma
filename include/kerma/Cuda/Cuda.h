#ifndef KERMA_CUDA_CUDA_H
#define KERMA_CUDA_CUDA_H


#include <string>

namespace kerma
{

/// Represents a 'side' for CUDA. That is, either HOST or DEVICE
/// Usually used to mark stuff like depending on which IR file (host, device) was used
enum class CudaSide {
  Unknown,
  HOST,
  DEVICE
};

/// Nvidia GPU Architectures
enum class CudaArch {
  Unknown,
  sm_30, sm_32,        // Kepler
  sm_35,
  sm_50, sm_52, sm_53, // Maxwell
  sm_60, sm_61, sm_62, // Pascal
  sm_70, sm_72,        // Volta
  sm_75                // Turing
};

/// Cuda Compute Capabilities
enum class CudaCompute {
  Unknown,
  cc_30, cc_32,        // Kepler
  cc_35,
  cc_37,
  cc_50, cc_52, cc_53, // Maxwell
  cc_60, cc_61, cc_62, // Pascal
  cc_70, cc_72,        // Volta
  cc_75                // Turing
};

/*
 * @brief Get a string value for a Cuda Compute Capability
 *
 * @param [in] c A Cuda Compute Capability
 * @param [in] full Get a string with a full description
 */
std::string getCudaComputeToStr(CudaCompute c, bool full=false);

/*
 * @brief Get a string value of an Nvidia GPU Architecture
 *
 * @param [in] arch An Nvidia Architecture
 */
std::string getCudaArchToStr(CudaArch arch);

/*
 * @brief Get the name of an Nvidia GPU Architecture
 *
 * @param [in] arch An Nvidia architecture
 */
std::string getCudaArchName(CudaArch arch);

/*
 * @brief Get the name of the GPU Architecture from a Compute Capability
 *
 * @param [in] cc A Compute Capability
 */
std::string getCudaArchName(CudaCompute cc);

/*
 * @brief Get an Nvidia Arch from a string
 *
 * @param [in] archName the name of the Nvidia Arch e.g sm_30, sm_52 etc
 */
CudaArch getCudaArch(const std::string& archName);

/*
 * @brief Get a string version of CudaSide
 *
 * @param [in] side a cuda side. e.g CudaSide::HOST
 */
std::string getCudaSideToStr(CudaSide side);

/*
 * @brief Get a CudaSide from a string
 *
 * @param [in] side the name of the CudaSide e.g 'host' or 'HOST'
 */
CudaSide getCudaSide(const std::string& side);

} /* NAMESPACE kerma */

#endif