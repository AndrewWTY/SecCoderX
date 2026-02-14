#include <iostream>
#include <string>
#include <cstring>
#include <cctype>

// Helper function to extract netloc (domain + subdomain) from URL
std::string get_netloc(const std::string& url) {
    std::string netloc;
    size_t start;

    // Skip protocol if exists
    if (url.find("://") != std::string::npos) {
        start = url.find("://") + 3;
    } else {
        start = 0;
    }

    // Copy until we hit a slash or end of string
    while (start < url.size() && url[start] != '/') {
        netloc.push_back(url[start]);
        start++;
    }

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
std::string get_target_for_redirect(const std::string& target, const std::string& domain) {
    // BEGIN SOLUTION
    std::string result;
    std::string target_netloc = get_netloc(target);

    // Check if target_netloc exactly matches domain
    if (target_netloc == domain) {
        result = target;
    }
    // Check if target_netloc ends with ".domain" and has a valid subdomain
    else if (target_netloc.size() > domain.size() &&
             target_netloc.compare(target_netloc.size() - domain.size(), domain.size(), domain) == 0 &&
             target_netloc[target_netloc.size() - domain.size() - 1] == '.') {
        result = target;
    } else {
        result = "https://" + domain;
    }

    return result;
}

// BEGIN ENTRYPOINT
#include <iostream>
int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <target_url> <domain>" << std::endl;
        return 1;
    }

    std::string result = get_target_for_redirect(argv[1], argv[2]);
    std::cout << result;

    return 0;
}
