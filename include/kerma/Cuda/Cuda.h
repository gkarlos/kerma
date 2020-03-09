#ifndef KERMA_CUDA_CUDA_H
#define KERMA_CUDA_CUDA_H


#include <string>

namespace kerma
{

enum class CudaVersion : unsigned char
{
  CUDA_70,
  CUDA_75,
  CUDA_80,
  CUDA_90,
  CUDA_91,
  CUDA_92,
  CUDA_100,
  CUDA_101,
  CUDA_102
};

/// Represents a 'side' for CUDA. That is, either HOST or DEVICE
/// Usually used to mark stuff like depending on which IR file (host, device) was used
enum class CudaSide : unsigned char
{
  HOST,
  DEVICE,
  Unknown,
};

/// Nvidia GPU Architectures
enum class CudaArch : unsigned char
{
  sm_10 = 10, sm_11, sm_12, sm_13, // Tesla
  sm_20, sm_21,        // Fermi
  sm_30, sm_32,        // Kepler
  sm_35,
  sm_50, sm_52, sm_53, // Maxwell
  sm_60, sm_61, sm_62, // Pascal
  sm_70, sm_72,        // Volta
  sm_75,               // Turing
  sm_80,               // Ampere
  Unknown
};

/// Cuda Compute Capabilities
enum class CudaCompute : unsigned char
{
  cc_10 = 10, cc_11, cc_12, cc_13, // Tesla
  cc_20, cc_21,        // Fermi
  cc_30, cc_32,        // Kepler
  cc_35,
  cc_37,
  cc_50, cc_52, cc_53, // Maxwell
  cc_60, cc_61, cc_62, // Pascal
  cc_70, cc_72,        // Volta
  cc_75,               // Turing
  cc_80,               // Ampere
  Unknown
};


const CudaCompute MIN_SUPPORTED_CUDA_COMPUTE = CudaCompute::cc_30;
const CudaCompute MAX_SUPPORTED_CUDA_COMPUTE = CudaCompute::cc_75;

const CudaArch MIN_SUPPORTED_CUDA_ARCH = CudaArch::sm_30;
const CudaArch MAX_SUPPORTED_CUDA_ARCH = CudaArch::sm_75;

/*
 * Check if a CudaCompute is supported
 */
bool isSupportedCudaCompute(CudaCompute cc);

/*
 * Check if a CudaArch is supported
 */
bool isSupportedCudaArch(CudaArch arch);

/*
 * @brief Get a string value for a Cuda Compute Capability
 *
 * @param [in] c A Cuda Compute Capability
 * @param [in] full Get a string with a full description
 */
std::string getCudaComputeStr(CudaCompute c, bool full=false);

/*
 * @brief Get a string value of an Nvidia GPU Architecture
 *
 * @param [in] arch An Nvidia Architecture
 */
std::string getCudaArchStr(CudaArch arch);

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
 * @brief Get an Nvidia Arch from a float value
 *
 * @param [in] arch value of the Nvidia Arch e.g 3.0 -> sm_30
 */
CudaArch getCudaArch(float arch);

/*
 * @brief Get a string version of CudaSide
 *
 * @param [in] side a cuda side. e.g CudaSide::HOST
 */
std::string getCudaSideStr(CudaSide side);

/*
 * @brief Get a CudaSide from a string
 *
 * @param [in] side the name of the CudaSide e.g 'host' or 'HOST'
 */
CudaSide getCudaSide(const std::string& side);

} /* NAMESPACE kerma */

#endif