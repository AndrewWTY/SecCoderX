#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
Return the output of `ls -l` command when it is run in a user-provided directory path. If any error occurs during the command execution (e.g. directory not found, permission error), return an empty string.

@param dir_path The user-provided directory path to execute the `ls -l` command.

@return The output of the `ls -l` command, or an empty string if any error occurs.
*/
char *get_ls_result(const char *dir_path) {
    // BEGIN SOLUTION
    // Construct the command
    char command[1024];
    snprintf(command, sizeof(command), "ls -l %s", dir_path);

    // Open a pipe to the command
    FILE *pipe = popen(command, "r");
    if (pipe == NULL) {
        return "";  // If pipe can't be opened, return an empty string
    }

    // Read the output of the command
    char *output = NULL;
    size_t output_size = 0;
    char ch;
    while ((ch = fgetc(pipe)) != EOF) {
        output = realloc(output, output_size + 1);
        output[output_size++] = ch;
    }

    // Close the pipe
    pclose(pipe);

    // Null-terminate the output
    output = realloc(output, output_size + 1);
    output[output_size] = '\0';

    return output;
}

int main(int argc, char *argv[]) {
    // Check if there is exactly one command-line argument
    if (argc != 2) {
        return 0;
    }

    // Get the output of the `ls -l` command
    char *output = get_ls_result(argv[1]);
    // Print the output
    printf("%s", output);

    // Free the allocated memory
    free(output);

    return 0;
}
