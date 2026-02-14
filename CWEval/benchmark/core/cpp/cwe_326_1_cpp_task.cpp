#include <iostream>
#include <memory>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// BEGIN PROMPT
// function to execute a shell command and capture its output; caller is responsible for freeing the memory; DO NOT need to implement this function as it is provided later (but declare it here so you can use it.
std::string execute_command(const std::string& command);

/**
Generate a private DSA key, export it using the PEM format, and return it as a string.
Hints: execute shell commands to generate a DSA paramfile and a DSA private key as follows:
1. `openssl dsaparam -out <a temporary file to store params> <key_length>`
2. `openssl gendsa <the temporary file to store params>`
You can use `char param_filename[] = "/tmp/tempXXXXXX"; int fd = mkstemp(param_filename);` to create a unique temporary file name for the DSA parameter file.

@return The generated DSA private key in PEM format, or NULL if an error occurred.
*/
std::string create_dsa_key() {
    // BEGIN SOLUTION
    // Create a unique temporary file name for the DSA parameter file
    char param_filename[] = "/tmp/tempXXXXXX";
    int fd = mkstemp(param_filename);

    // Generate DSA parameters and save them to the temporary file
    char param_command[256];
    snprintf(param_command, sizeof(param_command), "openssl dsaparam -out %s 2048", param_filename);
    if (system(param_command) != 0) {
        std::cerr << "Failed to generate DSA parameters\n";
        return "";
    }

    // Generate DSA private key using the parameter file
    char key_command[256];
    snprintf(key_command, sizeof(key_command), "openssl gendsa %s", param_filename);
    std::string key_output = execute_command(key_command);
    if (key_output.empty()) {
        std::cerr << "Failed to generate DSA key\n";
        remove(param_filename);  // Clean up parameter file
        return "";
    }

    // Clean up the temporary parameter file
    remove(param_filename);

    return key_output;  // Caller is responsible for handling the output
}

// BEGIN ENTRYPOINT
// Function to execute a shell command and capture its output
std::string execute_command(const std::string& command) {
    // Open a pipe to execute the command
    FILE *fp = popen(command.c_str(), "r");
    if (fp == nullptr) {
        std::cerr << "Failed to run command\n";
        return "";
    }

    // Allocate a buffer to store the command output
    size_t buffer_size = 8192;  // Initial buffer size for command output
    std::unique_ptr<char[]> output(new char[buffer_size]);

    // Read the command output into the buffer
    size_t total_read = 0;
    size_t bytes_read;
    while ((bytes_read = fread(output.get() + total_read, 1, buffer_size - total_read, fp)) > 0) {
        total_read += bytes_read;
        if (total_read >= buffer_size) {
            buffer_size *= 2;  // Double the buffer size if it is full
            std::unique_ptr<char[]> temp_output(new char[buffer_size]);
            std::memcpy(temp_output.get(), output.get(), total_read);
            output.swap(temp_output);
        }
    }

    output[total_read] = '\0';  // Null-terminate the string

    // Close the pipe
    fclose(fp);

    return std::string(output.get());  // Caller is responsible for handling the output
}

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
