#ifndef KERMA_TOOLS_KERMAD_SESSION_H
#define KERMA_TOOLS_KERMAD_SESSION_H

#include "Options.h"
#include <string>

namespace kerma {
namespace kermad {



class Session {
private:
  unsigned int ID;
  struct Options& Options;
  void setInput(const std::string& SourceDir,
                const std::string& Source);
  void createWorkingDir();

public:
  Session(struct Options& Options,
          const std::string& Dir,
          const std::string& Source);

  ~Session();

  std::string WorkingDirName;
  std::string WorkingDir;
  std::string SourceDir;
  std::string Source;
  std::string CompileDb;
  std::string SourcePath;
  std::string CompileDbPath;

  unsigned int getID() const { return ID; }
  std::string getSource() const { return Source; }
  std::string getSourceDir() const { return Source; }
  std::string getSourcePath() const { return SourcePath; }
  std::string getCompileDb() const { return CompileDb; }
  std::string getCompileDbPath() const { return CompileDbPath; }

  std::string getWorkingDir() const { return WorkingDir; }

};

}
}

#endif // KERMA_TOOLS_KERMAD_SESSION_H