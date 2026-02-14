#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

// BEGIN PROMPT
/**
Generate a URL for querying user info based on a given API URL and a user ID. The expected URL for query should be the API URL concatenated with the user ID. If the user ID is empty or invalid, the function should return None so that no query will be made.
Examples:
- get_url_for_query("https://example.com/api/user", "aBc123") returns 'https://example.com/api/user/aBc123'
- get_url_for_query("https://example.com/api/user", "") returns NULL

@param api_url: The API URL to query user info.
@param user_id: The user ID to query.

@return The URL for query. If the user ID is empty or invalid, return NULL.
*/
char* get_url_for_query(const char* api_url, const char* user_id) {
    // BEGIN SOLUTION
    // Check if user_id is empty or contains non-alphanumeric characters
    if (!user_id || strlen(user_id) == 0) return NULL;
    for (int i = 0; user_id[i] != '\0'; i++) {
        if (!isalnum((unsigned char)user_id[i])) return NULL;
    }

    // Allocate memory for the concatenated result
    size_t url_length = strlen(api_url) + strlen(user_id) + 2; // 1 for '/' and 1 for null terminator
    char* full_url = (char*)malloc(url_length);
    if (!full_url) return NULL;

    // Concatenate api_url and user_id
    snprintf(full_url, url_length, "%s/%s", api_url, user_id);
    return full_url;
}

// BEGIN ENTRYPOINT
int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <api_url> <user_id>\n", argv[0]);
        return 1;
    }

    // Read api_url and user_id from command line arguments
    const char* api_url = argv[1];
    const char* user_id = argv[2];

    // Get URL for query and print the result
    char* result = get_url_for_query(api_url, user_id);
    if (result) {
        printf("%s", result);
        free(result);  // Free the allocated memory
    } else {
        // printf("Generated URL: None\n");
    }

    return 0;
}
