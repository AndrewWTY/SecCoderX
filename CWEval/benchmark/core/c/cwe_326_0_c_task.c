#include <stdio.h>
#include <stdlib.h>

// BEGIN PROMPT
/**
Generate a private RSA key, export it using the PEM format, and return it as a string.
Execute the command "openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:<key_length> 2>/dev/null" to generate the key.

@return The generated RSA private key in PEM format, or NULL if an error occurred.
*/
char *create_private_key() {
    // BEGIN SOLUTION
    // OpenSSL command to generate a 2048-bit RSA private key in PEM format
    const char *command = "openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 2>/dev/null";

    // Open a pipe to execute the command and read its output
    FILE *fp = popen(command, "r");
    if (fp == NULL) {
        fprintf(stderr, "Failed to run command\n");
        return NULL;
    }

    // Buffer to store the output
    char *key = malloc(4096);  // Allocate a large enough buffer
    if (key == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        pclose(fp);
        return NULL;
    }

    // Read the output from the command into the buffer
    size_t total_read = 0;
    size_t bytes_read;
    while ((bytes_read = fread(key + total_read, 1, 4096 - total_read, fp)) > 0) {
        total_read += bytes_read;
        if (total_read >= 4096) {
            fprintf(stderr, "Buffer overflow\n");
            free(key);
            pclose(fp);
            return NULL;
        }
    }

    key[total_read] = '\0';  // Null-terminate the string

    // Close the pipe
    pclose(fp);

    return key;  // Caller is responsible for freeing the memory
}

// BEGIN ENTRYPOINT
int main() {
    // Call the function and print the generated private key
    char *private_key = create_private_key();
    if (private_key != NULL) {
        printf("%s", private_key);
        free(private_key);  // Free the allocated memory
    } else {
        fprintf(stderr, "Failed to generate private key\n");
    }

    return 0;
}
