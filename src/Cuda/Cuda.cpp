#include <kerma/Cuda/Cuda.h>

namespace kerma
{

std::string getCudaComputeToStr(CudaCompute c, bool full)
{
  switch (c) {
  case CudaCompute::cc_30:
    return full? "CudaCompute Capability 3.0" : "cc_30";
  case CudaCompute::cc_32:
    return full? "CudaCompute Capability 3.2" : "cc_32";
  case CudaCompute::cc_35:
    return full? "CudaCompute Capability 3.5" : "cc_35";
  case CudaCompute::cc_37:
    return full? "CudaCompute Capability 3.7" : "cc_37";
  case CudaCompute::cc_50:
    return full? "CudaCompute Capability 5.0" : "cc_50";
  case CudaCompute::cc_52:
    return full? "CudaCompute Capability 5.2" : "cc_52";
  case CudaCompute::cc_53:
    return full? "CudaCompute Capability 5.3" : "cc_53";
  case CudaCompute::cc_60:
    return full? "CudaCompute Capability 6.0" : "cc_60";
  case CudaCompute::cc_61:
    return full? "CudaCompute Capability 6.1" : "cc_61";
  case CudaCompute::cc_62:
    return full? "CudaCompute Capability 6.2" : "cc_62";
  case CudaCompute::cc_70:
    return full? "CudaCompute Capability 7.0" : "cc_70";
  case CudaCompute::cc_72:
    return full? "CudaCompute Capability 7.2" : "cc_72";
  case CudaCompute::cc_75:
    return full? "CudaCompute Capability 7.5" : "cc_75";
  default:
    return full? "Unknown CudaCompute Capability" : "Unknown";
  }
}

std::string getCudaArchToStr(CudaArch CudaArch)
{
  switch ( CudaArch) {
  case CudaArch::sm_30:
    return "sm_30";
  case CudaArch::sm_32:
    return "sm_32";
  case CudaArch::sm_35:
    return "sm_35";
  case CudaArch::sm_50:
    return "sm_50";
  case CudaArch::sm_52:
    return "sm_52";
  case CudaArch::sm_53:
    return "sm_53";
  case CudaArch::sm_60:
    return "sm_60";
  case CudaArch::sm_61:
    return "sm_61";
  case CudaArch::sm_62:
    return "sm_62";
  case CudaArch::sm_70:
    return "sm_70";
  case CudaArch::sm_72:
    return "sm_72";
  case CudaArch::sm_75:
    return "sm_75";
  default:
    return "Unknown Architecture";
  }
}

std::string getCudaArchName(CudaArch arch)
{
  switch ( arch) {
  case CudaArch::sm_30:
  case CudaArch::sm_32:
  case CudaArch::sm_35:
    return "Kepler";
  case CudaArch::sm_50:
  case CudaArch::sm_52:
  case CudaArch::sm_53:
    return "Maxwell";
  case CudaArch::sm_60:
  case CudaArch::sm_61:
  case CudaArch::sm_62:
    return "Pascal";
  case CudaArch::sm_70:
  case CudaArch::sm_72:
    return "Volta";
  case CudaArch::sm_75:
    return "Turing";
  default:
    return "Unknown";
  }
}

std::string getCudaArchName(CudaCompute cc)
{
  switch ( cc) {
  case CudaCompute::cc_30:
  case CudaCompute::cc_32:
  case CudaCompute::cc_35:
  case CudaCompute::cc_37:
    return "Kepler";
  case CudaCompute::cc_50:
  case CudaCompute::cc_52:
  case CudaCompute::cc_53:
    return "Maxwell";
  case CudaCompute::cc_60:
  case CudaCompute::cc_61:
  case CudaCompute::cc_62:
    return "Pascal";
  case CudaCompute::cc_70:
  case CudaCompute::cc_72:
  case CudaCompute::cc_75:
    return "Turing";
  default:
    return "Unknown";
  }
}

CudaArch getCudaArch(const std::string& arch) {
  if ( arch == "sm_30")
    return CudaArch::sm_30;
  else if ( arch == "sm_32")
    return CudaArch::sm_32;
  else if ( arch == "sm_35")
    return CudaArch::sm_35;
  else if ( arch == "sm_50")
    return CudaArch::sm_50;
  else if ( arch == "sm_52")
    return CudaArch::sm_52;
  else if ( arch == "sm_53")
    return CudaArch::sm_53;
  else if ( arch == "sm_60")
    return CudaArch::sm_60;
  else if ( arch == "sm_61")
    return CudaArch::sm_61;
  else if ( arch == "sm_62")
    return CudaArch::sm_62;
  else if ( arch == "sm_70")
    return CudaArch::sm_70;
  else if ( arch == "sm_72")
    return CudaArch::sm_72;
  else if ( arch == "sm_75")
    return CudaArch::sm_75;
  else
    return CudaArch::Unknown;
}

std::string getCudaSideToStr(CudaSide side)
{
  switch (side) {
  case CudaSide::HOST:
    return "host";
  case CudaSide::DEVICE:
    return "device";
  default:
    return "unknown";
  }
}

CudaSide getCudaSide(const std::string& side)
{
  if ( side.compare("host") == 0 || side.compare("Host") == 0 || side.compare("HOST") == 0 )
    return CudaSide::HOST;
  if ( side.compare("device") == 0 || side.compare("Device") == 0 || side.compare("DEVICE") == 0)
    return CudaSide::DEVICE;
  return CudaSide::Unknown;
}


} /* NAMESPACE kerma */