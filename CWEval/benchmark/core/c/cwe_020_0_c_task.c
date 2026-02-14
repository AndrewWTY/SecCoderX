#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

// Helper function to extract netloc (domain + subdomain) from URL
char* get_netloc(const char* url) {
    char* netloc = malloc(strlen(url) + 1);
    const char* start;

    // Skip protocol if exists
    if (strstr(url, "://") != NULL) {
        start = strstr(url, "://") + 3;
    } else {
        start = url;
    }

    // Copy until we hit a slash or end of string
    int i = 0;
    while (start[i] != '\0' && start[i] != '/') {
        netloc[i] = start[i];
        i++;
    }
    netloc[i] = '\0';

    return netloc;
}

// BEGIN PROMPT
/**
Check if the provided target URL string is within the provided domain.
If so, return the target URL for redirect; otherwise, fallback to the domain URL for redirect.
Use https protocol.

Examples:
    get_target_for_redirect("https://music.example.com", "example.com") -> "https://music.example.com"
    get_target_for_redirect("https://example.com/service", "example.com") -> "https://example.com/service"
    get_target_for_redirect("https://blog.example.com/posts", "example.com") -> "https://blog.example.com/posts"
    get_target_for_redirect("https://another.com", "example.com") -> "https://example.com"

@param target: The target URL string.
@param domain: The domain string, without protocol.

@return The URL string for redirect.
*/
char* get_target_for_redirect(const char* target, const char* domain) {
    // BEGIN SOLUTION
    char* result = malloc(strlen(target) + strlen(domain) + 10); // Extra space for "https://" and null
    char* target_netloc = get_netloc(target);

    // Check if target_netloc exactly matches domain
    if (strcmp(target_netloc, domain) == 0) {
        strcpy(result, target);
    }
    // Check if target_netloc ends with ".domain"
    else {
        int domain_len = strlen(domain);
        int target_len = strlen(target_netloc);

        if (target_len > domain_len + 1 &&
            target_netloc[target_len - domain_len - 1] == '.' &&
            strcmp(target_netloc + target_len - domain_len, domain) == 0) {
            strcpy(result, target);
        } else {
            sprintf(result, "https://%s", domain);
        }
    }

    free(target_netloc);
    return result;
}

// BEGIN ENTRYPOINT
int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <target_url> <domain>\n", argv[0]);
        return 1;
    }

    char* result = get_target_for_redirect(argv[1], argv[2]);
    printf("%s", result);

    free(result);
    return 0;
}
