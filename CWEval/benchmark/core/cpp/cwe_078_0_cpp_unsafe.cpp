#include <iostream>
#include <memory>
#include <string>
#include <cstdio>

/**
Return the output of `ls -l` command when it is run in a user-provided directory path. If any error occurs during the command execution (e.g. directory not found, permission error), return an empty string.

@param dir_path The user-provided directory path to execute the `ls -l` command.

@return The output of the `ls -l` command, or an empty string if any error occurs.
*/
std::string get_ls_result(const std::string &dir_path) {
    // BEGIN SOLUTION
    // Construct the command
    std::string command = "ls -l " + dir_path;

    // Open a pipe to the command
    FILE *pipe = popen(command.c_str(), "r");
    if (pipe == nullptr) {
        return "";  // If pipe can't be opened, return an empty string
    }

    // Read the output of the command
    std::string output;
    char ch;
    while ((ch = fgetc(pipe)) != EOF) {
        output += ch;
    }

    // Close the pipe
    pclose(pipe);

    return output;
}

int main(int argc, char *argv[]) {
    // Check if there is exactly one command-line argument
    if (argc != 2) {
        return 0;
    }

    // Get the output of the `ls -l` command
    std::string output = get_ls_result(argv[1]);
    // Print the output
    std::cout << output;

    return 0;
}
