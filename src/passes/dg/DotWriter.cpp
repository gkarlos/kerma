#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <kerma/passes/dg/Dot.h>
#include <kerma/Support/FileSystem.h>

#include <string>
#include <system_error>

using namespace llvm;
using namespace kerma;

// #define DEBUG_TYPE "DotWriter"

DotWriter::DotWriter(const std::string& filename)
: DotWriter(get_cwd(), filename)
{}

DotWriter::DotWriter(const std::string& dir, const std::string& file)
: directory_(( dir == "" || dir == "." || dir == "./")? get_cwd() : dir), 
  filename_(file.empty()? "unnamed_dotfile.dot": file)
{}

std::string DotWriter::getDirectory() { return directory_; }
std::string DotWriter::getFilename() { return filename_; }

void
DotWriter::write(std::set<DotNode> &nodes, std::set<DotEdge> &edges)
{
  errs() << "WRITTING DOT: " << nodes.size() << ", " << edges.size() << "\n";
  std::error_code err;
  std::string outfile(directory_ + "/" + filename_);
  // LLVM_DEBUG(errs() << "Writting file: " << outfile << "\n");

  llvm::raw_fd_ostream out(outfile, err);
  if ( err) {
    // LLVM_DEBUG(errs() << "Could not open file\n");
    return;
  }

  out << "digraph __dg__{\n";

  for ( auto node : nodes)
    out << node.getName() << " " << node.getValue() << "\n";

  for ( auto edge : edges) {
    errs() << "Writting edge: " << edge << "\n";
    out << edge.getValue() << "\n";
  }
    
  out << "}\n";

  out.close(); 
}



