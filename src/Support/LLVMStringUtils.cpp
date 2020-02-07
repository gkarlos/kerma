//
// Created by gkarlos on 13-01-20.
//

#include <kerma/Support/LLVMStringUtils.h>

#include <llvm/IR/Instructions.h>

namespace kerma
{

std::string
getStringFromLLVMValue( const llvm::Value *v)
{
  if ( v == nullptr)
    return "";
  std::string str;
  llvm::raw_string_ostream rso(str);
  v->print( rso);
  return str;
}

std::string
getStringFromLLVMInstr( const llvm::Instruction *I)
{
  if ( I == nullptr)
    return "";
  std::string str;
  llvm::raw_string_ostream rso(str);
  I->print( rso);
  return str;
}


std::string
getDbgLocString( const llvm::Instruction *I)
{
  auto &dbg = I->getDebugLoc();
  if ( dbg)
    return std::to_string(dbg.getLine()) + ":" + std::to_string(dbg.getCol());
  else
    return "";
}


std::string& ltrim(std::string &s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
    return !std::isspace(ch);
  }));
  return s;
}

std::string& ltrim(std::string &&s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
    return !std::isspace(ch);
  }));
  return s;
}

}
