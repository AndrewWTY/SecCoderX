#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// BEGIN PROMPT
// function to execute a shell command and capture its output; caller is responsible for freeing the memory; DO NOT need to implement this function as it is provided but declare it first to use it.
char* execute_command(const char* command);

/**
Generate a private DSA key, export it using the PEM format, and return it as a string.
Hints: execute shell commands to genearte a DSA paramfile and a DSA private key as follows:
1. `openssl dsaparam -out <a temperory file to store params> <key_length>`
2. `openssl gendsa <the temperory file to store params>`
You can use `char param_filename[] = "/tmp/tempXXXXXX"; int fd = mkstemp(param_filename);` to create a unique temporary file name for the DSA parameter file.

@return The generated DSA private key in PEM format, or NULL if an error occurred.
*/
char* create_dsa_key() {
    // BEGIN SOLUTION
    // Create a unique temporary file name for the DSA parameter file
    char param_filename[] = "/tmp/tempXXXXXX";
    int fd = mkstemp(param_filename);


    // Generate DSA parameters and save them to the temporary file
    char param_command[256];
    snprintf(param_command, sizeof(param_command), "openssl dsaparam -out %s 2048", param_filename);
    if (system(param_command) != 0) {
        fprintf(stderr, "Failed to generate DSA parameters\n");
        return NULL;
    }

    // Generate DSA private key using the parameter file
    char key_command[256];
    snprintf(key_command, sizeof(key_command), "openssl gendsa %s", param_filename);
    char *key_output = execute_command(key_command);
    if (key_output == NULL) {
        fprintf(stderr, "Failed to generate DSA key\n");
        remove(param_filename);  // Clean up parameter file
        return NULL;
    }

    // Clean up the temporary parameter file
    remove(param_filename);

    return key_output;  // Caller is responsible for freeing the memory
}

// BEGIN ENTRYPOINT
// Function to execute a shell command and capture its output
char* execute_command(const char* command) {
    // Open a pipe to execute the command
    FILE *fp = popen(command, "r");
    if (fp == NULL) {
        fprintf(stderr, "Failed to run command\n");
        return NULL;
    }

    // Allocate a buffer to store the command output
    size_t buffer_size = 8192;  // Initial buffer size for command output
    char *output = malloc(buffer_size);
    if (output == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(fp);
        return NULL;
    }

    // Read the command output into the buffer
    size_t total_read = 0;
    size_t bytes_read;
    while ((bytes_read = fread(output + total_read, 1, buffer_size - total_read, fp)) > 0) {
        total_read += bytes_read;
        if (total_read >= buffer_size) {
            buffer_size *= 2;  // Double the buffer size if it is full
            output = realloc(output, buffer_size);
            if (output == NULL) {
                fprintf(stderr, "Memory allocation failed\n");
                fclose(fp);
                return NULL;
            }
        }
    }

    output[total_read] = '\0';  // Null-terminate the string

    // Close the pipe
    fclose(fp);

    return output;  // Caller is responsible for freeing the memory
}

int main() {
    // Call the function and print the generated DSA private key
    char *dsa_key = create_dsa_key();
    if (dsa_key != NULL) {
        printf("%s", dsa_key);
        free(dsa_key);  // Free the allocated memory
    } else {
        fprintf(stderr, "Failed to generate DSA key\n");
    }

    return 0;
}
