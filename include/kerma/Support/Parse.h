#ifndef KERMA_SUPPORT_PARSE_H
#define KERMA_SUPPORT_PARSE_H

#include <vector>
#include <string>

namespace kerma {

/// Parse a delimited separated string and return a vector with the results
///
/// @param s The original string
/// @param delim Delimiter
/// @param cb An optional transform callback. Will be called on each extracted substr
std::vector<std::string> parseDelimStr( const std::string& s,
                                        char delim,
                                        std::string(*cb)(const std::string& subStr)=nullptr);

}

#endif // KERMA_SUPPORT_PARSE_H