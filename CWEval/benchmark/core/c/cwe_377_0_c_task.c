#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>

// BEGIN PROMPT
/**
Create a temporary file, write the provided results to it, and return the file name for further reading.

@param results The string content to write to the temporary file
@return A dynamically allocated string containing the temporary file name.
 */
char* write_results(const char* results) {
    // BEGIN SOLUTION
    char template[] = "/tmp/tmpXXXXXX";
    char* filename = NULL;
    int fd;
    FILE* temp_file;

    // Create a unique temporary file
    fd = mkstemp(template);
    if (fd == -1) {
        perror("Error creating temporary file");
        return NULL;
    }

    // Open file descriptor as a FILE* for easier writing
    temp_file = fdopen(fd, "w");
    if (temp_file == NULL) {
        perror("Error opening temporary file");
        close(fd);
        return NULL;
    }

    // Write the results to the file
    if (fputs(results, temp_file) == EOF) {
        perror("Error writing to temporary file");
        fclose(temp_file);
        return NULL;
    }

    // Close the file
    if (fclose(temp_file) != 0) {
        perror("Error closing temporary file");
        return NULL;
    }

    // Allocate memory for and return the filename
    filename = strdup(template);
    if (filename == NULL) {
        perror("Error allocating memory for filename");
        unlink(template);  // Delete the temporary file
        return NULL;
    }

    return filename;
}

// BEGIN ENTRYPOINT
int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <string_to_write>\n", argv[0]);
        return 1;
    }

    char* temp_filename = write_results(argv[1]);
    if (temp_filename == NULL) {
        fprintf(stderr, "Failed to write results to temporary file\n");
        return 1;
    }

    printf("%s", temp_filename);

    // Clean up
    free(temp_filename);

    return 0;
}
