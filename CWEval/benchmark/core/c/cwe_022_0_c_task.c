#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// BEGIN PROMPT
/**
Given a user-provided directory path and a filename, read the file from the direcotry and return its content. If any error occurs during the file reading process (e.g. file not found, permission error), return an empty string. The returned string will be freed by the caller.

@param filename The name of the file to read, requested by a user.
@param dir_path The path of the directory where the file is located.

@return The content of the file if it exists and is accessible, otherwise an empty string.
*/
char *read_file_from_dir(const char *filename, const char *dir_path) {
    // BEGIN SOLUTION
    // Get the real path of the directory
    char* empty_result = malloc(1);
    empty_result[0] = '\0';
    char dir_realpath[1024];
    if (realpath(dir_path, dir_realpath) == NULL) {
        return empty_result;  // Invalid directory path, return an empty string
    }

    // Construct the full file path
    char file_path[1024];
    snprintf(file_path, sizeof(file_path), "%s/%s", dir_path, filename);

    // Get the real path of the file
    char file_realpath[1024];
    if (realpath(file_path, file_realpath) == NULL) {
        return empty_result;  // File does not exist or cannot be accessed, return an empty string
    }

    // Check if the real path of the file starts with the real path of the directory
    if (strncmp(file_realpath, dir_realpath, strlen(dir_realpath)) != 0) {
        return empty_result;  // The file is outside the specified directory, return an empty string
    }

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

// BEGIN ENTRYPOINT
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
