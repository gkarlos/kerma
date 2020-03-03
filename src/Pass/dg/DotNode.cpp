#include "llvm/IR/Argument.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Casting.h"
#include <kerma/passes/dg/Dot.h>
#include <kerma/Support/LLVMStringUtils.h>

#include <algorithm>
#include <string>
#include <utility>
#include <iostream>

namespace kerma
{

static const std::string DEFAULT_VALUE = "";
static const std::string DEFAULT_SHAPE = "";
static const std::string DEFAULT_LABEL = "";
static const std::string DEFAULT_COLOR = "";
static const std::string DEFAULT_FILL_COLOR = "";

DotNode::DotNode(const std::string& name) 
: DotNode(name, nullptr)
{}

DotNode::DotNode(const std::string& name, llvm::Value *V)
: DotNode(name, DEFAULT_LABEL, DEFAULT_SHAPE, DEFAULT_FILL_COLOR, DEFAULT_COLOR)
{
  setLLVMValue(V);
}

DotNode::DotNode( const std::string& name, const std::string& label, 
                  const std::string& shape, const std::string& fillCol, 
                  const std::string& borderCol)
: valueChanged_(true),
  name_(name),
  lastValue_(DEFAULT_VALUE),
  label_(label), 
  shape_(shape), 
  fillColor_(fillCol), 
  borderColor_(borderCol),
  LLVMValue_(nullptr),
  ty_(DotNode::Ty::NONE)
{
  getValue();
}

DotNode&
DotNode::setName(const std::string& name) 
{ 
  name_.assign(name); 
  valueChanged_ = true;
  return *this;
}

DotNode&
DotNode::setLabel(const std::string& label)
{
  label_.assign(label);
  valueChanged_ = true;
  return *this;
}

DotNode&
DotNode::setShape(const std::string& shape)
{
  shape_.assign(shape);
  valueChanged_ = true;
  return *this;
}

DotNode&
DotNode::setFillColor(const std::string& color)
{
  fillColor_.assign(color);
  valueChanged_ = true;
  return *this;
}

DotNode&
DotNode::setBorderColor(const std::string& color)
{
  borderColor_.assign(color);
  valueChanged_ = true;
  return *this;
}

DotNode&
DotNode::setLLVMValue(llvm::Value *V)
{
  LLVMValue_ = V;
  ty_ = DotNode::getTyFromLLVMValue(*V);
  label_.assign(V == nullptr? DEFAULT_LABEL : ltrim(getStringFromLLVMValue(V)));
  valueChanged_ = true;
  return *this;
}

DotNode::Ty DotNode::getTy()
{
  return ty_;
}

const std::string& DotNode::getTyStr()
{
  return DotNode::getTyStr(ty_);
}

std::string DotNode::getValue() 
{
  if ( !valueChanged_)
    return lastValue_;
  
  lastValue_.clear();
  lastValue_.append("[label=\"").append(label_).append("\"");

  if ( shape_.compare(DEFAULT_SHAPE) != 0)
    lastValue_.append(" shape=\"").append(shape_).append("\"");
  
  if ( borderColor_.compare(DEFAULT_COLOR) != 0)
    lastValue_.append(" color=\"").append(borderColor_).append("\"");
  
  //TODO fillColor

  lastValue_.append("]");

  valueChanged_ = false;
  return lastValue_;
}

std::string DotNode::getName() { return name_; }
std::string DotNode::getLabel() { return label_; }
std::string DotNode::getShape() { return shape_; }
std::string DotNode::getFillColor() { return fillColor_; }
std::string DotNode::getBorderColor() { return borderColor_; }
llvm::Value *DotNode::getLLVMValue() { return LLVMValue_; }

DotNode& DotNode::operator=(const DotNode& other) 
{
  if ( this != &other) {
    name_.assign(other.name_);
    lastValue_.assign(other.lastValue_);
    label_.assign(other.label_);
    shape_.assign(other.shape_);
    fillColor_.assign(other.fillColor_);
    borderColor_ = other.borderColor_;
    LLVMValue_ = other.LLVMValue_;
    ty_ = other.ty_;
    valueChanged_ = true;
  }
  return *this;
}

bool DotNode::operator<(const DotNode& other) const
{
  return name_ < other.name_;
}

bool DotNode::operator==(const DotNode& other)
{
  if ( valueChanged_)
    getValue();
  
  return !name_.compare(other.name_) && !lastValue_.compare(other.lastValue_);
}

bool DotNode::operator!=(const DotNode &other)
{
  return !operator==(other);
}


DotNode::Ty
DotNode::getTyFromLLVMValue(llvm::Value &V)
{
  if ( auto *I = llvm::dyn_cast<llvm::Instruction>(&V)) {
    Ty type = Ty::INSTRUCTION;

    if ( auto *L = llvm::dyn_cast<llvm::LoadInst>(I))
      type = Ty::LOAD;
    else if ( auto *S = llvm::dyn_cast<llvm::StoreInst>(I))
      type = Ty::STORE;
    else if ( auto *A = llvm::dyn_cast<llvm::AllocaInst>(I))
      type = Ty::ALLOCA;
    
    return type;
  }
  
  if ( auto *C = llvm::dyn_cast<llvm::Constant>(&V))
    return Ty::CONST;
  if ( auto *Arg = llvm::dyn_cast<llvm::Argument>(&V))
    return Ty::ARG;

  return Ty::NONE;
}

const std::string&
DotNode::getTyStr(Ty type)
{
  static const std::map<Ty, std::string> tyStrMap = {
    {Ty::LOAD,        "L"},
    {Ty::STORE,       "S"},
    {Ty::ALLOCA,      "a"},
    {Ty::INSTRUCTION, "I"},
    {Ty::CONST,       "C"},
    {Ty::ARG,         "A"},
    {Ty::VALUE,       "V"},
    {Ty::NONE,        "-"}
  };
  return tyStrMap.find(type)->second;
}

}