#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/**
Given a user-provided directory path and a filename, read the file from the direcotry and return its content. If any error occurs during the file reading process (e.g. file not found, permission error), return an empty string.

@param filename The name of the file to read, requested by a user.
@param dir_path The path of the directory where the file is located.

@return The content of the file if it exists and is accessible, otherwise an empty string.
*/
char *read_file_from_dir(const char *filename, const char *dir_path) {
    // BEGIN SOLUTION
    // Construct the file path
    char* empty_result = malloc(1);
    empty_result[0] = '\0';
    char file_path[1024];
    snprintf(file_path, sizeof(file_path), "%s/%s", dir_path, filename);

    // Open the file for reading
    FILE *file = fopen(file_path, "r");
    if (file == NULL) {
        return empty_result;  // If file can't be opened, return an empty string
    }

    // Read the contents of the file
    char *content = NULL;
    size_t content_size = 0;
    char ch;
    while ((ch = fgetc(file)) != EOF) {
        content = realloc(content, content_size + 1);
        content[content_size++] = ch;
    }

    // Close the file
    fclose(file);

    // Null-terminate the content
    content = realloc(content, content_size + 1);
    content[content_size] = '\0';

    free(empty_result);
    return content;
}

int main(int argc, char *argv[]) {
    // Check if there are exactly two command-line arguments
    if (argc != 3) {
        return 0;
    }

    // Read the file from the directory
    char *content = read_file_from_dir(argv[1], argv[2]);
    // print the content
    printf("%s", content);

    // Free the allocated memory
    free(content);

    return 0;
}
