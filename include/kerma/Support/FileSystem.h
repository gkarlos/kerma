//
// Created by gkarlos on 21-01-20.
//

#ifndef KERMA_STATIC_ANALYSIS_FILESYSTEM_H
#define KERMA_STATIC_ANALYSIS_FILESYSTEM_H

#include <string>

namespace kerma
{

/*
 * @brief Retrieve the cwd
 */
std::string get_cwd();

/*
 * @brief Retrieve an absolute path from a relative one
 */
std::string get_realpath(const std::string& relpath);

bool fileExists(const std::string& name);

}
#endif // KERMA_STATIC_ANALYSIS_FILESYSTEM_H
