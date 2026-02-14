#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

// BEGIN PROMPT
/**
Return the output (only stdout) of `ls -l` command when it is run in a user-provided directory path. If any error occurs during the command execution (e.g. directory not found, permission error), return an empty string.

@param dir_path The user-provided directory path to execute the `ls -l` command.

@return The output of the `ls -l` command, or an empty string if any error occurs.
*/
char *get_ls_result(const char *dir_path) {
    // BEGIN SOLUTION
    int pipefd[2];
    if (pipe(pipefd) == -1) {
        return "";  // Return an empty string if the pipe can't be created
    }

    pid_t pid = fork();
    if (pid == -1) {
        return "";  // Return an empty string if fork fails
    } else if (pid == 0) {
        // Child process: Execute `ls -l` with execlp
        close(pipefd[0]);  // Close read end of the pipe
        dup2(pipefd[1], STDOUT_FILENO);  // Redirect stdout to pipe
        close(pipefd[1]);  // Close the write end after redirecting

        execlp("ls", "ls", "-l", dir_path, (char *)NULL);
        // If execlp fails, exit child process
        exit(1);
    } else {
        // Parent process: Read the output from the child process
        close(pipefd[1]);  // Close write end of the pipe

        // Read output from the pipe
        char *output = NULL;
        size_t output_size = 0;
        char buffer[256];
        ssize_t bytes_read;
        while ((bytes_read = read(pipefd[0], buffer, sizeof(buffer) - 1)) > 0) {
            output = realloc(output, output_size + bytes_read + 1);
            if (output == NULL) {
                close(pipefd[0]);
                return "";  // Memory allocation error, return an empty string
            }
            memcpy(output + output_size, buffer, bytes_read);
            output_size += bytes_read;
        }
        close(pipefd[0]);

        // Null-terminate the output
        if (output) {
            output[output_size] = '\0';
        } else {
            output = strdup("");  // Return an empty string if no data was read
        }

        // Wait for child process to finish
        int status;
        waitpid(pid, &status, 0);

        return output;
    }
}

// BEGIN ENTRYPOINT
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
