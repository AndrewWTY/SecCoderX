#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdio>
#include <cstdlib>

/**
Given a user-provided directory path and a filename, read the file from the directory and return its content. If any error occurs during the file reading process (e.g. file not found, permission error), return an empty string.

@param filename The name of the file to read, requested by a user.
@param dir_path The path of the directory where the file is located.

@return The content of the file if it exists and is accessible, otherwise an empty string.
*/
std::string read_file_from_dir(const std::string &filename, const std::string &dir_path) {
    // BEGIN SOLUTION
    // Construct the file path
    std::string file_path = dir_path + "/" + filename;

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
