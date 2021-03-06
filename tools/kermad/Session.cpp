#include "Session.h"

#include <boost/filesystem.hpp>
#include <boost/filesystem/operations.hpp>
#include "kerma/Support/Log.h"
#include "kerma/Support/FileSystem.h"
#include "spdlog/spdlog.h"
#include <llvm/IR/LegacyPassManager.h>
#include <unistd.h>

#include <spdlog/fmt/fmt.h>

namespace kerma {
namespace kermad {

namespace fs = boost::filesystem;

static unsigned int ids = 0;

Session::Session(struct Options &Options,
                 const std::string& Dir,
                 const std::string& Source)
: Options(Options), ID(ids++) {
  setInput(Dir, Source);
  createWorkingDir();
  // Once a session is create we chdir to the
  // session's dir as our working directory
  chdir(WorkingDir.c_str());
  // Create the SourceInfoBuilder
  SIB = std::make_unique<SourceInfoBuilder>(getSourcePath());
  Log::info(LOG_SEP);
  Log::info("✔ Session {}, CWD: {}", ID, WorkingDir);
}

Session::~Session() {
  chdir(Options.WorkingDir.c_str());
  Log::info("✘ Session {}, CWD: {}", ID, Options.WorkingDir);
  Log::info(LOG_SEP);
}

void
Session::createWorkingDir() {
  fs::create_directory(WorkingDir);
  Options.addCleanupDir(WorkingDir);
}

void
Session::setInput(const std::string &SourceDir,
                  const std::string &Source) {
  this->SourceDir = SourceDir;
  this->Source = Source;
  this->SourcePath = (fs::path(SourceDir) / fs::path(Source)).string();
  this->CompileDbPath = (fs::path(SourceDir) / fs::path("compile_commands.json")).string();
  this->WorkingDirName = Options.InvocationID + "-sess" + std::to_string(ID);
  this->WorkingDir = (fs::path(Options.WorkingDir) / fs::path(WorkingDirName)).string();
}

}
}