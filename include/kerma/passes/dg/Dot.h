
#ifndef KERMA_STATIC_ANALYSIS_DG_DOT_H
#define KERMA_STATIC_ANALYSIS_DG_DOT_H

#include "llvm/IR/Argument.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/raw_ostream.h"
#include <ostream>
#include <string>
#include <set>

namespace kerma
{

class Dot;
class DotNode;
class DotEdge;
class DotWriter;



/*
 * Helper class that representing a Dot node
 */
class DotNode 
{
public:
  enum class Ty {
    /* Node associated with an llvm::LoadInst */
    LOAD,
    /* Node associated with an llvm::StoreInst */
    STORE,
    /* Node associated with an llvm::AllocaInst */
    ALLOCA,
    /* Node associated with an llvm::Instruction not listed above */
    INSTRUCTION, 
    /* Node associated with a function argument */
    ARG,
    /* Node associated with an llvm::Constant */
    CONST,
    /* Node associated with an llvm::Value not listed above */
    VALUE,
    /* Absence of Type. The node is not associated with any relevant Value */
    NONE
  };

  DotNode(const std::string& name);
  DotNode(const std::string& name, 
          llvm::Value *V);
  DotNode(const std::string& name, 
          const std::string& label, 
          const std::string& shape, 
          const std::string& fillCol, 
          const std::string& borderCol);
  ~DotNode()=default;

  /*
   * Set the name of this node. Triggers (lazy) name change 
   */
  DotNode& setName(const std::string& name);

  /* 
   * Set the label (textual content) of the node. Triggers (lazy) value change
   */
  DotNode& setLabel(const std::string& label);
  
  /* 
   * Set the shape of the node. Triggers (lazy) value change
   */
  DotNode& setShape(const std::string& shape);
  
  /* 
   * Set the color of the node. Triggers (lazy) value change
   */
  DotNode& setFillColor(const std::string& color);

  /* 
   * Set the border color of the node. Triggers (lazy) value change
   */
  DotNode& setBorderColor(const std::string& color);

  /* 
   * Assign an LLVM #Value to the node. Triggers (lazy) value change
   */
  DotNode& setLLVMValue(llvm::Value *V);

  /*
   * Retrieve the label (textual content) of the node
   */
  std::string getLabel();

  /*
   * Retrieve the shape of the node
   */
  std::string getShape();

  /*
   * Retrieve the color of the node
   */
  std::string getFillColor();

  /*
   * Retrieve the border color of the node
   */
  std::string getBorderColor();

  /*
   * Retrieve the LLVM Value associated with the node
   */
  llvm::Value *getLLVMValue();

  /*
   * Retrieve the type of of node
   */
  Ty getTy();

  /*
   * Retrieve a String representation of the type of this node
   */
  const std::string& getTyStr();

  /*
   * Retrieve the name of a dot node. That is, the value by which
   * the node can be referred to.
   */
  std::string getName();

  /*
   * Retrieve a string describing the node (color, label etc)
   *        
   * @detail - A node is described by: <name> <value> pair in dot.
   *           Subsequently, edges can be defined by: <name> -> <name>
   *         - Values are stored and returned by subsequent calls
   *           until the node is changed and a new value is computed
   */
  std::string getValue();

  /// Static

  /*
   * Retrieve a DotNodeTy from an LLVM Value
   */
  static DotNode::Ty getTyFromLLVMValue(llvm::Value &V);

  /*
   * Get a String representation of a type
   */
  static const std::string& getTyStr(Ty type) ;


  /// Operators
  DotNode& operator=(const DotNode& other); //copy
  bool operator<(const DotNode& other) const;
  bool operator==(const DotNode& other);
  bool operator!=(const DotNode& other);

  friend std::ostream& operator<<(std::ostream& os, DotNode& node)
  { 
    os << DotNode::getTyStr(node.ty_) << "( " << node.name_ << ", " << node.getValue() << " )";
    return os;
  }

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os, DotNode& node)
  { 
    os << DotNode::getTyStr(node.ty_) << "( " << node.name_ << ", " << node.getValue() << " )";
    return os;
  }
  
private:
  bool valueChanged_;
  std::string name_;
  std::string lastValue_;
  std::string label_;
  std::string shape_;
  std::string fillColor_;
  std::string borderColor_;
  llvm::Value *LLVMValue_;
  Ty ty_;
};


/*
 * Wrapper class representing an edge between two Dot nodes
 */
class DotEdge
{
public:
  DotEdge(DotNode &src, DotNode &tgt);
  DotEdge(const DotEdge& other);
  ~DotEdge()=default;
  void setSource(DotNode &src);
  void setTarget(DotNode &tgt);
  DotNode &getSource();
  DotNode &getTarget();
  std::string getValue();

  DotEdge& operator=(DotEdge& other); //copy
  
  bool operator< (const DotEdge& other) const;
  bool operator==(const DotEdge& other);
  bool operator!=(const DotEdge& other);

  friend std::ostream& operator<<(std::ostream& os, DotEdge& e)
  {
    os << e.getSource().getName() << " -> " << e.getTarget().getName();
    return os;
  }

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream& ros, DotEdge& e)
  {
    ros << e.getSource().getName() << " -> " << e.getTarget().getName();
    return ros;
  }
private:
  DotNode &src_;
  DotNode &tgt_;
};


/* 
 * Containes a number of factory methods
 */
class Dot
{
public:
  /// inst
  static DotNode createNode(void);
  static DotNode createNode(llvm::LoadInst &load);
  static DotNode createNode(llvm::StoreInst &store);
  static DotNode createNode(llvm::AllocaInst &alloca);
  static DotNode createNode(llvm::Instruction &instr);
  /// non-inst
  static DotNode createNode(llvm::Constant &constant);
  static DotNode createNode(llvm::Argument &argument);
  static DotNode createNode(llvm::Value &V);
};



class DotWriter
{
public:
  explicit DotWriter(const std::string& filename);
  DotWriter(const std::string& dir, const std::string& filename);

  void write(std::set<DotNode> &nodes, std::set<DotEdge> &edges);
  
  std::string getDirectory();
  std::string getFilename();


private:
  const std::string filename_;
  const std::string directory_;
};

} /// NAMESPACE kerma

#endif /// KERMA_STATIC_ANALYSIS_DG_DOT_H