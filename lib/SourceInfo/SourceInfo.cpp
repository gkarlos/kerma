#include "kerma/SourceInfo/SourceInfo.h"
#include "kerma/SourceInfo/SourceRange.h"

#include "llvm/Support/Path.h"

namespace kerma {

SourceInfo::SourceInfo() : SourceInfo("", SourceRange::Unknown, "")
{}

SourceInfo::SourceInfo( const std::string& path,
                        const SourceRange& range,
                        const std::string& text)
: Path(path), Range(range), Text(text)
{
  splitPath();
}

std::string SourceInfo::getFilename() const { return Filename; }

std::string SourceInfo::getDirectory() const { return Directory; }

std::string SourceInfo::getPath() const { return Path; }

SourceInfo& SourceInfo::setPath(std::string &path) { 
  Path = path;
  splitPath();
  return *this;
}

SourceInfo& SourceInfo::setPath(const char *path) {
  Path = path;
  splitPath();
  return *this;
}

SourceRange SourceInfo::getRange() const { return Range; }

std::string& SourceInfo::getText() { return Text; }

SourceInfo& SourceInfo::setText(std::string &text) {
  Text = text;
  return *this;
}

SourceInfo& SourceInfo::setText(const char *text) {
  Text = text;
  return *this;
}

bool SourceInfo::operator==(const SourceInfo &other) const {
  return Path == other.Path && Range == other.Range && Text == other.Text;
}

bool SourceInfo::operator!=(const SourceInfo &other) const {
  return !(*this == other);
}

void SourceInfo::splitPath() {
  Filename = llvm::sys::path::filename(Path);
  Directory = llvm::sys::path::parent_path(Path);
}

} // namespace kerma