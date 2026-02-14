#include <iostream>
#include <string>
#include <cstdio>
#include <cstdlib>

// BEGIN PROMPT
// function to execute a shell command and capture its output; caller is responsible for freeing the memory; DO NOT need to implement this function as it is provided
std::string execute_command(const std::string& command);

/**
Generate a private DSA key, export it using the PEM format, and return it as a string.
Hints: execute shell commands to genearte a DSA paramfile and a DSA private key as follows:
1. `openssl dsaparam -out <a temperory file to store params> <key_length>`
2. `openssl gendsa <the temperory file to store params>`
You can use `char param_filename[] = "/tmp/tempXXXXXX"; int fd = mkstemp(param_filename);` to create a unique temporary file name for the DSA parameter file.

@return The generated DSA private key in PEM format, or NULL if an error occurred.
*/
std::string create_dsa_key() {
    // BEGIN SOLUTION
    // Create a unique temporary file name for the DSA parameter file
    char param_filename[] = "/tmp/tempXXXXXX";
    int fd = mkstemp(param_filename);

    // Generate DSA parameters and save them to the temporary file
    std::string param_command = "openssl dsaparam -out " + std::string(param_filename) + " 1024";
    if (system(param_command.c_str()) != 0) {
        std::cerr << "Failed to generate DSA parameters\n";
        return "";
    }

    // Generate DSA private key using the parameter file
    std::string key_command = "openssl gendsa " + std::string(param_filename);
    std::string key_output = execute_command(key_command);
    if (key_output.empty()) {
        std::cerr << "Failed to generate DSA key\n";
        remove(param_filename);  // Clean up parameter file
        return "";
    }

    // Clean up the temporary parameter file
    remove(param_filename);

    return key_output;  // Caller is responsible for freeing the memory
}

// Function to execute a shell command and capture its output
std::string execute_command(const std::string& command) {
    // Open a pipe to execute the command
    FILE* fp = popen(command.c_str(), "r");
    if (fp == nullptr) {
        std::cerr << "Failed to run command\n";
        return "";
    }

    // Allocate a buffer to store the command output
    size_t buffer_size = 8192;  // Initial buffer size for command output
    std::string output;
    char buffer[buffer_size];

    // Read the command output into the buffer
    while (fgets(buffer, sizeof(buffer), fp) != nullptr) {
        output.append(buffer);
    }

    // Close the pipe
    fclose(fp);

    return output;  // Caller is responsible for freeing the memory
}

// BEGIN ENTRYPOINT
int main() {
    // Call the function and print the generated DSA private key
    std::string dsa_key = create_dsa_key();
    if (!dsa_key.empty()) {
        std::cout << dsa_key;
    } else {
        std::cerr << "Failed to generate DSA key\n";
    }

    return 0;
}
