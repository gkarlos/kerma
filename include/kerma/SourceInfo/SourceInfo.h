#ifndef KERMA_SOURCEINFO_SOURCEINFO_H
#define KERMA_SOURCEINFO_SOURCEINFO_H

#include "kerma/SourceInfo/SourceRange.h"
#include <string>

namespace kerma {

class SourceInfo {
private:
  std::string Filename;
  std::string Directory;
  std::string Path;
  std::string Text;
  SourceRange Range;


public:
  SourceInfo();
  SourceInfo( const std::string& path, 
              const SourceRange& range=SourceRange::Unknown, 
              const std::string& text="");

  std::string getFilename() const;

  std::string getDirectory() const;

  std::string getPath() const;
  SourceInfo& setPath(std::string& path);
  SourceInfo& setPath(const char *path);

  SourceRange getRange() const;
  SourceInfo& setRange(SourceRange& range) const;

  std::string& getText();
  SourceInfo& setText(std::string& text);
  SourceInfo& setText(const char* text);

  bool operator==(const SourceInfo &other) const;
  bool operator!=(const SourceInfo &other) const;


private:
  void splitPath();

};

} // namespace kerma

#endif // KERMA_SOURCEINFO_SOURCEINFO_H