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
    char tmp_name[L_tmpnam];
    char* filename = NULL;
    int fd;
    FILE* temp_file;

    if (tmpnam(tmp_name) != NULL) {
        filename = strdup(tmp_name);
        if (filename == NULL) {
            perror("Error allocating memory for filename");
            return NULL;
        }
        temp_file = fopen(filename, "w");
        if (temp_file == NULL) {
            perror("Error opening temporary file");
            free(filename);
            return NULL;
        }
        if (fputs(results, temp_file) == EOF) {
            perror("Error writing to temporary file");
            fclose(temp_file);
            free(filename);
            return NULL;
        }
        if (fclose(temp_file) != 0) {
            perror("Error closing temporary file");
            free(filename);
            return NULL;
        }
        return filename;
    } else {
        return NULL;
    }
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
