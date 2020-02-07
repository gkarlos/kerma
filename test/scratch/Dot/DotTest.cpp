#include <kerma/passes/dg/Dot.h>

#include <iostream>

int main(int argc, char **argv)
{
  kerma::DotNode node("node1", "this is some LLVM node", "box", "", "red");
  std::cout << node << "\n";
  std::cout << kerma::Dot::createNode() << "\n";
}