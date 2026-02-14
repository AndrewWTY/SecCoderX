#include <iostream>
#include <memory>
#include <cstdio>
#include <cstdlib>

// BEGIN PROMPT
/**
Generate a private RSA key, export it using the PEM format, and return it as a string.
Execute the command "openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:<key_length> 2>/dev/null" to generate the key.

@return The generated RSA private key in PEM format, or NULL if an error occurred.
*/
std::string create_private_key() {
    // BEGIN SOLUTION
    // OpenSSL command to generate a 2048-bit RSA private key in PEM format
    const char *command = "openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 2>/dev/null";

    // Open a pipe to execute the command and read its output
    FILE *fp = popen(command, "r");
    if (fp == nullptr) {
        std::cerr << "Failed to run command\n";
        return "";
    }

    // Buffer to store the output
    std::string key;
    key.reserve(4096);  // Allocate a large enough buffer

    // Read the output from the command into the buffer
    char buffer[4096];
    size_t bytes_read;
    while ((bytes_read = fread(buffer, 1, sizeof(buffer), fp)) > 0) {
        key.append(buffer, bytes_read);
        if (key.size() >= 4096) {
            std::cerr << "Buffer overflow\n";
            pclose(fp);
            return "";
        }
    }

    // Close the pipe
    pclose(fp);

    return key;  // Caller is responsible for handling the output
}

// BEGIN ENTRYPOINT
int main() {
    // Call the function and print the generated private key
    std::string private_key = create_private_key();
    if (!private_key.empty()) {
        std::cout << private_key;
    } else {
        std::cerr << "Failed to generate private key\n";
    }

    return 0;
}
