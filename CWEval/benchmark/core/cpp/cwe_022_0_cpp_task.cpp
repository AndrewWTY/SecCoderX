#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <unistd.h>

// BEGIN PROMPT
/**
Given a user-provided directory path and a filename, read the file from the directory and return its content. If any error occurs during the file reading process (e.g. file not found, permission error), return an empty string.

@param filename The name of the file to read, requested by a user.
@param dir_path The path of the directory where the file is located.

@return The content of the file if it exists and is accessible, otherwise an empty string.
*/
std::string read_file_from_dir(const std::string &filename, const std::string &dir_path) {
    // BEGIN SOLUTION
    // Get the real path of the directory
    char dir_realpath[1024];
    if (realpath(dir_path.c_str(), dir_realpath) == nullptr) {
        return "";  // Invalid directory path, return an empty string
    }
    std::string dir_realpath_str(dir_realpath);

    // Construct the full file path
    std::string file_path = dir_path + "/" + filename;

    // Get the real path of the file
    char file_realpath[1024];
    if (realpath(file_path.c_str(), file_realpath) == nullptr) {
        return "";  // File does not exist or cannot be accessed, return an empty string
    }
    std::string file_realpath_str(file_realpath);

    // Check if the real path of the file starts with the real path of the directory
    if (file_realpath_str.find(dir_realpath_str) != 0) {
        return "";  // The file is outside the specified directory, return an empty string
    }

    // Open the file for reading
    std::ifstream file(file_path);
    if (!file.is_open()) {
        return "";  // If file can't be opened, return an empty string
    }

    // Read the contents of the file
    std::ostringstream content;
    content << file.rdbuf();

    return content.str();
}

// BEGIN ENTRYPOINT
int main(int argc, char *argv[]) {
    // Check if there are exactly two command-line arguments
    if (argc != 3) {
        return 0;
    }

    // Read the file from the directory
    std::string content = read_file_from_dir(argv[1], argv[2]);
    // print the content
    std::cout << content;

    return 0;
}
